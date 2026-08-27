//! SMT-aware core topology: which logical CPUs share a physical core.
//!
//! # Why this exists
//!
//! [`crate::decode_affinity`] discovers *NUMA nodes*. That is one of the two
//! topology facts a spinning thread pool needs, and it is the less important
//! one on a single-node host. The other is **which logical CPUs are siblings on
//! one physical core**, because a busy-waiting worker does not share a core
//! gracefully: two spinning SMT siblings interleave in the front end and each
//! runs at roughly half rate, while one spinning sibling next to one *working*
//! sibling steals issue slots from real arithmetic.
//!
//! This is not hypothetical on this project. `docs/benchmarks/`
//! `2026-08-15-cpu-ep-vs-ort-attention-moe.md` §26.2(b) records a bistable
//! result — the same command measuring 0.030 ms and 0.441 ms on consecutive
//! runs — traced to 15 pinned spinning workers confined to CPUs 0-15, which on
//! that host is **8 physical cores** (siblings are adjacent: `cpu0`/`cpu1` =
//! core 0). A blocktime sweep over 500/200/50/20/5/0 us produced ratios from
//! 0.233 to 72.3 with no monotone trend, which is the signature of a scheduling
//! pathology rather than a tuning parameter. §27.1 lists "SMT-aware spin
//! capping" as blocker #1 for the entire t=8/16 column, and this module is that
//! discovery step.
//!
//! # The compact-mask result, and how much of it survived
//!
//! This section used to read: "It does not spread work across physical cores.
//! Pinning one thread per physical core across `0,2,…,30` on the same host
//! measured **worse** (0.133 ms vs 0.079 ms) because the working set fits one
//! CCX's L3, so the compact mask is correct and stays." That conclusion has been
//! superseded for the persistent SPMD decode pool, and the reason is worth
//! recording rather than deleting.
//!
//! What the 0.133/0.079 experiment did not separate is the **dispatcher**. The
//! inline dispatcher spin-waits on the completion counters, so once every
//! physical core holds a worker it has nowhere to run but some worker's SMT
//! sibling — and that worker becomes the straggler the whole barrier waits for,
//! on every op. Pinning only the dispatcher thread and changing nothing else
//! measured **2.1x** (19.58 ms/token on a CPU inside the worker set against
//! 8.77 ms on a free core), with ~600 involuntary context switches per token in
//! the contended case. A bare spread therefore pays the locality cost *and*
//! creates a straggler, which is a fair description of the losing arm.
//!
//! [`crate::decode_spmd`] can now opt in to spreading one worker per physical
//! core **and** reserving a core for the dispatcher. Quiet-host A/B against the
//! compact layout, four launches per arm with an A/A null agreeing to 0.17%:
//! 3.2% faster on llama and **29% less CPU per token**, and the dispatcher-yield
//! counter falls 5.00 to 0.00 per token, which is the straggler mechanism
//! closing. That remains an explicit dedicated-host mode, not the default.
//!
//! Two parts of the old conclusion **do** survive and should not be re-litigated
//! without measurement:
//!
//! * **Locality is real.** The spread crosses the L3 boundary (`0-15` and
//!   `16-31` are separate 32 MiB domains on this host) and that costs something;
//!   it is outweighed here, not absent.
//! * **Compactness is robust to co-tenants.** Because the compact layout uses
//!   half the machine, a co-tenant can be placed on the idle half. Measured with
//!   four DRAM-bandwidth hogs: compact 4.54 ms/token against spread 5.03--6.26.
//!   With eight hogs covering both halves the ranking inverts (compact 15.92
//!   against spread 6.08), because then the compact layout has 8 contended cores
//!   and the spread has 16. Spread is the right explicit mode for a dedicated
//!   host or cpuset; it is not free on a shared box and is not the default.
//!
//! Capping how many *spinning* workers live inside a mask remains a separate and
//! still-useful lever.
//!
//! # Portability
//!
//! * **Linux** — `/sys/devices/system/cpu/cpuN/topology/core_cpus_list`, falling
//!   back to the older `thread_siblings_list`. Both are cpu-list syntax
//!   (`"0,16"`, `"0-1"`).
//! * **Windows** — `GetLogicalProcessorInformationEx(RelationProcessorCore)`,
//!   whose `PROCESSOR_RELATIONSHIP` names each core's logical processors as
//!   group affinities. CPU indices use the same `group * 64 + bit` encoding as
//!   [`crate::decode_affinity`].
//! * **macOS** — `sysctlbyname("hw.physicalcpu"/"hw.logicalcpu")`. The scope
//!   directive for Apple work is macOS arm64, which has no SMT, so the derived
//!   grouping is one logical CPU per core and the cap is the identity.
//! * **Anything else** — `None`, and every consumer degrades to "no cap".
//!
//! Detection is fallible everywhere and every caller treats `None` as "do not
//! cap", never as "cap to 1".

use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Which logical CPUs share a physical core.
///
/// Invariants, established by [`CoreTopology::from_sibling_groups`] and relied
/// on by every accessor: each group is sorted ascending and non-empty, no CPU
/// appears in two groups, and the groups themselves are ordered by their lowest
/// CPU index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoreTopology {
    cores: Vec<Vec<usize>>,
}

impl CoreTopology {
    /// Build from raw sibling groups, normalising them into the documented
    /// invariants: sort within a group, drop empties, drop a CPU already claimed
    /// by an earlier group (the sysfs files list the *same* group once per
    /// sibling, so raw reads are full of duplicates), then order by first CPU.
    ///
    /// This is the seam the unit tests drive: every policy function below is
    /// pure over a `CoreTopology`, so SMT policy is testable on a host with no
    /// SMT, and a host with SMT can be tested for the non-SMT case.
    pub fn from_sibling_groups(groups: impl IntoIterator<Item = Vec<usize>>) -> Self {
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        let mut cores: Vec<Vec<usize>> = Vec::new();
        for group in groups {
            let mut group: Vec<usize> = group.into_iter().filter(|c| !seen.contains(c)).collect();
            group.sort_unstable();
            group.dedup();
            if group.is_empty() {
                continue;
            }
            seen.extend(group.iter().copied());
            cores.push(group);
        }
        cores.sort_unstable_by_key(|group| group[0]);
        Self { cores }
    }

    /// Detect the running host's core topology, or `None` when this platform
    /// exposes none (which every caller reads as "do not cap").
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(target_vendor = "apple")]
        {
            Self::detect_apple()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_vendor = "apple")))]
        {
            None
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Option<Self> {
        let entries = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
        let mut cpus: Vec<usize> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            // Skip an unreadable entry rather than abandoning the walk: a `?`
            // here would let one odd name in the directory return `None` for
            // the whole machine, and `None` silently disables every placement
            // assertion in this crate.
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(index) = name.strip_prefix("cpu") else {
                continue;
            };
            // `cpufreq`, `cpuidle`, `cpu_capacity` and friends also start with
            // `cpu`; only a pure integer suffix names a logical CPU.
            let Ok(index) = index.parse::<usize>() else {
                continue;
            };
            cpus.push(index);
        }
        cpus.sort_unstable();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for cpu in cpus {
            let dir = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            // `core_cpus_list` is the modern name; `thread_siblings_list` is the
            // pre-5.3 spelling and is still what container images with older
            // kernels expose. A CPU that exposes neither is its own core.
            let list = std::fs::read_to_string(format!("{dir}/core_cpus_list"))
                .or_else(|_| std::fs::read_to_string(format!("{dir}/thread_siblings_list")))
                .ok();
            match list.as_deref().map(parse_cpu_list) {
                Some(siblings) if !siblings.is_empty() => groups.push(siblings),
                _ => groups.push(vec![cpu]),
            }
        }
        let topology = Self::from_sibling_groups(groups);
        (!topology.cores.is_empty()).then_some(topology)
    }

    #[cfg(target_os = "windows")]
    fn detect_windows() -> Option<Self> {
        let groups = windows_cores::processor_cores()?;
        let topology = Self::from_sibling_groups(groups);
        (!topology.cores.is_empty()).then_some(topology)
    }

    /// macOS reports counts, not a sibling map. Apple Silicon (the only Apple
    /// target in scope) has no SMT, so `logical == physical` and the derived
    /// grouping is one CPU per core — exactly the identity cap. If a host ever
    /// reports `logical > physical`, siblings are modelled as consecutive
    /// indices, which is the conventional enumeration and is only ever used to
    /// *count* cores here.
    #[cfg(target_vendor = "apple")]
    fn detect_apple() -> Option<Self> {
        let physical = apple_sysctl_usize(c"hw.physicalcpu")?;
        let logical = apple_sysctl_usize(c"hw.logicalcpu")?;
        if physical == 0 || logical < physical {
            return None;
        }
        let per_core = logical / physical;
        let groups = (0..physical)
            .map(|core| (0..per_core).map(|s| core * per_core + s).collect())
            .collect::<Vec<Vec<usize>>>();
        Some(Self::from_sibling_groups(groups))
    }

    /// The number of physical cores discovered.
    pub fn core_count(&self) -> usize {
        self.cores.len()
    }

    /// The number of logical CPUs discovered.
    pub fn logical_count(&self) -> usize {
        self.cores.iter().map(Vec::len).sum()
    }

    /// The sibling groups, ordered by lowest CPU index.
    pub fn cores(&self) -> &[Vec<usize>] {
        &self.cores
    }

    /// True when at least one physical core has more than one logical CPU.
    pub fn has_smt(&self) -> bool {
        self.cores.iter().any(|group| group.len() > 1)
    }

    /// The logical CPUs sharing `cpu`'s physical core, including `cpu` itself.
    pub fn siblings_of(&self, cpu: usize) -> Option<&[usize]> {
        self.cores
            .iter()
            .find(|group| group.contains(&cpu))
            .map(Vec::as_slice)
    }

    /// How many distinct physical cores the CPUs in `allowed` cover.
    ///
    /// This — not `allowed.len()` — is the number of threads that can spin
    /// without two of them sharing a core's front end. A CPU that is not in the
    /// discovered topology counts as its own core, so an incomplete sysfs view
    /// can only ever *over*-count, never silently collapse the cap to 1.
    pub fn physical_cores_within(&self, allowed: &[usize]) -> usize {
        let allowed: BTreeSet<usize> = allowed.iter().copied().collect();
        let mut cores = 0;
        let mut covered: BTreeSet<usize> = BTreeSet::new();
        for group in &self.cores {
            if group.iter().any(|cpu| allowed.contains(cpu)) {
                cores += 1;
                covered.extend(group.iter().copied());
            }
        }
        cores + allowed.iter().filter(|cpu| !covered.contains(cpu)).count()
    }

    /// One CPU per physical core within `allowed`, in ascending order.
    ///
    /// The leader of a core is its lowest allowed sibling, so a compact mask
    /// such as `0..16` on a host whose siblings are adjacent yields
    /// `0,2,4,…,14` — still inside the original mask, so this never widens the
    /// process's CPU set. Unknown CPUs are leaders of their own core.
    pub fn leaders_within(&self, allowed: &[usize]) -> Vec<usize> {
        let allowed_set: BTreeSet<usize> = allowed.iter().copied().collect();
        let mut leaders: Vec<usize> = Vec::new();
        let mut covered: BTreeSet<usize> = BTreeSet::new();
        for group in &self.cores {
            if let Some(&leader) = group.iter().find(|cpu| allowed_set.contains(cpu)) {
                leaders.push(leader);
                covered.extend(group.iter().copied());
            }
        }
        leaders.extend(
            allowed_set
                .iter()
                .copied()
                .filter(|cpu| !covered.contains(cpu)),
        );
        leaders.sort_unstable();
        leaders.dedup();
        leaders
    }
}

/// Parse a Linux cpu-list string such as `"0-3,8,10-11"` into CPU indices.
///
/// Deliberately a second copy of [`crate::decode_affinity`]'s parser rather than
/// a shared one: that one is `#[cfg(any(target_os = "linux", test))]` and
/// private, and duplicating twenty lines is cheaper than widening a module's
/// public surface for it. Both are covered by their own tests.
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
                && start <= end
                // A range is materialised, so a malformed one is an allocation
                // and not just a wrong answer: `0-4294967295` from a container
                // shim or an emulated sysfs would try to push four billion
                // entries. Nothing downstream can use a CPU id this large --
                // `physical_cores_within` intersects with the affinity mask --
                // so dropping the whole range is both safe and cheap.
                && end - start < MAX_CPU_LIST_SPAN
            {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Widest `a-b` run [`parse_cpu_list`] will expand. Comfortably above any real
/// machine (the largest shipping x86 socket is three digits of threads) and far
/// below anything that costs real memory.
#[cfg(any(target_os = "linux", test))]
const MAX_CPU_LIST_SPAN: usize = 1 << 16;

#[cfg(target_vendor = "apple")]
fn apple_sysctl_usize(name: &std::ffi::CStr) -> Option<usize> {
    let mut value: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `name` is a NUL-terminated C string, `value` is a live `i32` and
    // `len` truthfully reports its size; a null new-value pointer with length 0
    // is the documented read-only form.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

/// The host's core topology, detected once.
///
/// Cached because the consumers are pool-construction paths that may run per
/// session, and a Linux detection walks one directory plus one small file per
/// logical CPU.
pub fn host() -> Option<&'static CoreTopology> {
    static TOPOLOGY: OnceLock<Option<CoreTopology>> = OnceLock::new();
    TOPOLOGY.get_or_init(CoreTopology::detect).as_ref()
}

/// Whether this target has a core-topology backend at all.
///
/// Compile-time, and that is the whole point. Every placement assertion in this
/// crate used to be reached through `host()` returning `Some`, so a detection
/// regression converted them into no-ops instead of failures -- including the
/// anti-vacuity guard that exists to report exactly that, which was written
/// `if allowed_now >= 2 && core_topology::host().is_some()`. A guard predicated
/// on the runtime success of the thing whose failure it reports cannot fire.
///
/// Skipping is legitimate only where there is no backend to succeed, and that
/// is a property of the *target*, not of a runtime call, so it is answered by
/// `cfg!` and cannot silently become false on a host that should have answered.
#[cfg(test)]
pub(crate) const DETECTION_SUPPORTED: bool = cfg!(any(
    target_os = "linux",
    target_os = "windows",
    target_vendor = "apple"
));

/// Reason returned when a skip is genuinely justified. Stated rather than
/// silent, so a skipped placement check is visible in the assertion text.
#[cfg(test)]
pub(crate) const NO_BACKEND_REASON: &str = "core-topology detection has no backend on this target, so placement is unanswerable by \
     construction rather than by failure";

/// The fail-closed policy behind [`require_host_for_placement`], split out so
/// the mutation test can force a detection failure without mutating global
/// state that parallel tests share.
///
/// `detection_supported` is a parameter rather than a direct read of
/// [`DETECTION_SUPPORTED`] for the same reason: it lets one test assert the
/// panic branch and another assert the skip branch, on whichever host CI runs.
#[cfg(test)]
pub(crate) fn topology_or_fail_closed(
    detected: Option<&'static CoreTopology>,
    detection_supported: bool,
) -> Result<&'static CoreTopology, &'static str> {
    match (detected, detection_supported) {
        (Some(topology), _) => Ok(topology),
        (None, true) => panic!(
            "core-topology detection returned None on a target that supports it. Every \
             placement assertion in this crate is reached through this call, so treating \
             this as a skip would silently convert them -- and the anti-vacuity guard that \
             reports that conversion -- into no-ops that still pass. See #1792: a pool \
             reporting `realized=16 as_requested` while running on half the cores it \
             claimed is the defect; a checker that reports nothing is indistinguishable \
             from a checker that passes."
        ),
        (None, false) => Err(NO_BACKEND_REASON),
    }
}

/// Topology for placement assertions: fails closed wherever detection is
/// supported, and skips with an explicit reason only where it is not.
///
/// Placement *tests* must call this rather than [`host`]. `host` keeps its
/// `Option` because production callers must not panic on an undiscoverable
/// topology -- `planned_placement_is_one_worker_per_physical_core` answering `None` is
/// correct, since claiming `Some(true)` there would be a lie. The difference is
/// that a test skipping is a defect being missed, whereas production declining
/// to cap is the documented "never cap on a guess" policy.
#[cfg(test)]
pub(crate) fn require_host_for_placement() -> Result<&'static CoreTopology, &'static str> {
    topology_or_fail_closed(host(), DETECTION_SUPPORTED)
}

/// The variable a CI lane sets to declare that this crate's placement
/// falsifiers must actually execute there.
///
/// Modelled on `NXRT_REQUIRE_ORT_TESTS`, which the `cli-ort` job sets so that a
/// missing ORT reddens the one lane whose job is to run against ORT, while a
/// developer without ORT still gets a skip. The same asymmetry applies here:
/// no host is obliged to have two physical cores or a working
/// `sched_setaffinity`, but the lanes we point at this crate are, and until
/// something says so their absence is indistinguishable from a pass.
#[cfg(test)]
pub(crate) const REQUIRE_PLACEMENT_ENV: &str = "NXRT_REQUIRE_PLACEMENT_TESTS";

/// Whether the current lane has declared the placement falsifiers mandatory.
#[cfg(test)]
pub(crate) fn placement_tests_required() -> bool {
    std::env::var(REQUIRE_PLACEMENT_ENV).as_deref() == Ok("1")
}

/// The second placement capability, after detection: at least two physical
/// cores to place workers on.
///
/// Returns `true` when the check may proceed. When it may not, this panics in a
/// lane that declared the checks mandatory and otherwise states the skip on
/// stderr -- as opposed to the bare `return` these sites used before, which is
/// a silent pass.
///
/// Detection already fails closed through [`require_host_for_placement`], but
/// that only proves the topology is *readable*. A host that reads back one
/// physical core switches every "one worker per physical core" assertion off
/// while satisfying the anti-vacuity guard, because the guard asserts
/// `core_count() > 0`. The capabilities the assertions need and the capability
/// the guard checks are not the same set.
#[cfg(test)]
pub(crate) fn require_two_cores_for_placement(cores: &CoreTopology, what: &str) -> bool {
    if cores.core_count() >= 2 {
        return true;
    }
    assert!(
        !placement_tests_required(),
        "{REQUIRE_PLACEMENT_ENV}=1 but this host reports {} physical core(s), so `{what}` cannot \
         distinguish one worker per physical core from two workers sharing one, and would report \
         success without testing anything. Point this lane at a runner with two physical cores or \
         stop requiring placement tests on it.",
        cores.core_count()
    );
    eprintln!(
        "skipping {what}: {} physical core(s) cannot express \"one worker per core\" \
         distinguishably",
        cores.core_count()
    );
    false
}

/// The CPU layout a placement falsifier has to be able to build before it can
/// answer anything.
///
/// `require_two_cores_for_placement` asks what the *host* has. That is not the
/// same question as what this *process* may use: a `taskset`, cpuset or
/// container narrows the allowed set, and a check whose layout is
/// unrepresentable inside it returns without testing anything.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum PlacementLayout {
    /// Two SMT siblings of one physical core -- the *bad* layout a negative
    /// control must construct in order to report it as a defect.
    SharedCore,
    /// Two CPUs on different physical cores -- the good layout the positive arm
    /// needs, without which the negative arm above it is equally satisfied by a
    /// predicate that is simply always false.
    DistinctCores,
}

#[cfg(test)]
impl PlacementLayout {
    /// The CPUs realising this layout inside `allowed`, if it is representable
    /// there at all.
    pub(crate) fn within(
        self,
        cores: &CoreTopology,
        allowed: &BTreeSet<usize>,
    ) -> Option<Vec<usize>> {
        match self {
            Self::SharedCore => cores.cores().iter().find_map(|group| {
                let mut inside = group.iter().copied().filter(|cpu| allowed.contains(cpu));
                match (inside.next(), inside.next()) {
                    (Some(first), Some(second)) => Some(vec![first, second]),
                    _ => None,
                }
            }),
            Self::DistinctCores => {
                let cpus: Vec<usize> = cores
                    .cores()
                    .iter()
                    .filter_map(|group| group.iter().copied().find(|cpu| allowed.contains(cpu)))
                    .take(2)
                    .collect();
                (cpus.len() == 2).then_some(cpus)
            }
        }
    }

    /// Whether the *host* could supply this layout if nothing were narrowing the
    /// allowed set. This is the discriminator between a platform fact and a
    /// binding artifact, and it is why the skip cannot be a bare `return`.
    pub(crate) fn representable_on(self, cores: &CoreTopology) -> bool {
        match self {
            Self::SharedCore => cores.cores().iter().any(|group| group.len() >= 2),
            Self::DistinctCores => cores.core_count() >= 2,
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::SharedCore => "two SMT siblings of one physical core",
            Self::DistinctCores => "two CPUs on distinct physical cores",
        }
    }
}

/// The third placement capability, after detection and two physical cores: the
/// layout has to be representable inside the CPU set this process may use.
///
/// Returns the CPUs to use, or `None` when the check cannot run. `None` is a
/// panic in a lane that declared placement mandatory *and* whose host could
/// have supplied the layout -- i.e. when the only reason the check cannot run
/// is how the run was bound. A host that genuinely lacks the layout (no SMT, or
/// one physical core) is exempt, because failing there would be a false failure
/// and the pressure would be to delete the requirement rather than fix the lane.
///
/// `required` is a parameter rather than a direct call to
/// [`placement_tests_required`] for the same reason `detection_supported` is one
/// on [`topology_or_fail_closed`]: both states have to be assertable from a
/// multi-threaded test binary, where toggling the process-wide environment would
/// make unrelated placement tests observe whichever value won the race.
#[cfg(test)]
pub(crate) fn placement_cpus_or_fail_closed(
    layout: PlacementLayout,
    cores: &CoreTopology,
    allowed: &BTreeSet<usize>,
    required: bool,
    what: &str,
) -> Option<Vec<usize>> {
    if let Some(cpus) = layout.within(cores, allowed) {
        return Some(cpus);
    }
    let representable_on_host = layout.representable_on(cores);
    assert!(
        !(representable_on_host && required),
        "{REQUIRE_PLACEMENT_ENV}=1 and this host can express {} somewhere, but not inside the CPU \
         set this process may use ({allowed:?}), so `{what}` cannot build the layout it exists to \
         judge and would report success without testing anything. That is a property of how this \
         run was bound -- taskset, cpuset or container -- not of the platform: bind the lane to \
         CPUs that can express {}, or stop requiring placement tests on it.",
        layout.description(),
        layout.description()
    );
    if representable_on_host {
        eprintln!(
            "skipping {what}: this run is bound to {allowed:?}, which cannot express {}",
            layout.description()
        );
    } else {
        eprintln!(
            "skipping {what}: this host cannot express {} at all, so the layout is \
             unrepresentable rather than merely unreachable",
            layout.description()
        );
    }
    None
}

/// The number of physical cores the current process may actually run on.
///
/// Intersects the detected core topology with the process's allowed CPU set
/// (`sched_getaffinity` / `GetThreadGroupAffinity`), so a container, `taskset`,
/// or this crate's own `bound_process_to_decode_budget` confinement is reflected
/// rather than ignored. `None` means the answer is unknown — callers must treat
/// that as "do not cap", because capping on a guess is how a 32-thread host
/// silently becomes a 1-thread host.
pub fn allowed_physical_cores() -> Option<usize> {
    let topology = host()?;
    match crate::decode_affinity::allowed_cpus() {
        Some(allowed) => Some(topology.physical_cores_within(&allowed)),
        // The allowed set is unknown, so every discovered core is assumed
        // reachable. This is the same "do not guess a restriction" policy
        // `NumaTopology::restrict_to_allowed` applies.
        None => Some(topology.core_count()),
    }
    .filter(|&cores| cores > 0)
}

/// Cap `requested` spinning workers at one per physical core the process may
/// run on.
///
/// Returns `requested` unchanged when the topology is undiscoverable, when the
/// host has no SMT, or when the request already fits — so this is a no-op on
/// every non-SMT machine and can only ever reduce, never inflate, a thread
/// count. Never returns 0 for a non-zero request.
pub fn cap_spinning_workers(requested: usize) -> usize {
    match allowed_physical_cores() {
        Some(cores) => requested.min(cores).max(usize::from(requested > 0)),
        None => requested,
    }
}

#[cfg(target_os = "windows")]
mod windows_cores {
    use std::mem::size_of;

    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    /// Bits per processor group mask (`KAFFINITY` is 64-bit on x64/arm64).
    const GROUP_BITS: usize = 64;

    fn cpus_from_mask(group: u16, mask: usize) -> Vec<usize> {
        let base = group as usize * GROUP_BITS;
        (0..GROUP_BITS)
            .filter(|bit| mask & (1usize << bit) != 0)
            .map(|bit| base + bit)
            .collect()
    }

    /// One `Vec<usize>` of sibling logical CPUs per physical core, via
    /// `GetLogicalProcessorInformationEx(RelationProcessorCore)`.
    ///
    /// A `PROCESSOR_RELATIONSHIP` record carries `GroupCount` group affinities
    /// in a trailing flexible array, so a core spanning processor groups (rare,
    /// but legal) is read completely rather than truncated to its first group.
    pub(super) fn processor_cores() -> Option<Vec<Vec<usize>>> {
        let mut len: u32 = 0;
        // SAFETY: the null-buffer/zero-length form is the documented size query;
        // it only writes `len` and fails with ERROR_INSUFFICIENT_BUFFER.
        unsafe {
            GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len);
        }
        if len == 0 {
            return None;
        }
        let mut buffer = vec![0u8; len as usize];
        // SAFETY: `buffer` is a live `len`-byte allocation and we pass its true
        // length; on success the OS fills it with packed variable-length records.
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                buffer.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                &mut len,
            )
        };
        if ok == 0 {
            return None;
        }

        let mut cores: Vec<Vec<usize>> = Vec::new();
        let mut offset = 0usize;
        let end = len as usize;
        while offset + size_of::<u32>() * 2 <= end {
            // SAFETY: `offset` leaves at least the `Relationship`+`Size` header
            // inside the filled region; the struct is `Copy` and the buffer has
            // alignment 1, so an unaligned by-value read is the sound form.
            let record = unsafe {
                std::ptr::read_unaligned(
                    buffer.as_ptr().add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
                )
            };
            let size = record.Size as usize;
            if size == 0 || offset + size > end {
                break;
            }
            if record.Relationship == RelationProcessorCore {
                // SAFETY: the relationship tag says the `Processor` union arm is
                // the active one.
                let processor = unsafe { record.Anonymous.Processor };
                let group_count = processor.GroupCount as usize;
                let mut siblings = Vec::new();
                // `GroupMask` is declared as a 1-element array but is a trailing
                // flexible array of `GroupCount` entries; walk it by offset from
                // the record base so the bytes read stay inside `Size`.
                //
                // The offset is the field's position *within the record type* --
                // computing it as a delta between `&processor` (a by-value copy
                // read out of the union) and `&record` would subtract two
                // unrelated stack addresses and produce a garbage offset, which
                // fails the bounds check below and silently drops every core.
                let masks_offset = offset
                    + std::mem::offset_of!(
                        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                        Anonymous.Processor.GroupMask
                    );
                for i in 0..group_count {
                    let entry_offset = masks_offset
                        + i * size_of::<
                            windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY,
                        >();
                    if entry_offset
                        + size_of::<windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY>()
                        > offset + size
                    {
                        break;
                    }
                    // SAFETY: bounds-checked against this record's `Size` above;
                    // `GROUP_AFFINITY` is POD and read unaligned.
                    let affinity = unsafe {
                        std::ptr::read_unaligned(buffer.as_ptr().add(entry_offset)
                            as *const windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY)
                    };
                    siblings.extend(cpus_from_mask(affinity.Group, affinity.Mask as usize));
                }
                if !siblings.is_empty() {
                    cores.push(siblings);
                }
            }
            offset += size;
        }
        (!cores.is_empty()).then_some(cores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-core / 32-thread host with adjacent siblings: `cpu0`/`cpu1` share
    /// core 0. This is exactly the AMD EPYC 9V74 layout every CPU-EP benchmark
    /// in `docs/benchmarks/` was taken on.
    fn adjacent_smt(cores: usize) -> CoreTopology {
        CoreTopology::from_sibling_groups((0..cores).map(|c| vec![c * 2, c * 2 + 1]))
    }

    /// A host whose siblings are `cpu` and `cpu + n`, the other common
    /// enumeration (Intel, and AMD in some firmware modes).
    fn split_smt(cores: usize) -> CoreTopology {
        CoreTopology::from_sibling_groups((0..cores).map(|c| vec![c, c + cores]))
    }

    fn no_smt(cores: usize) -> CoreTopology {
        CoreTopology::from_sibling_groups((0..cores).map(|c| vec![c]))
    }

    #[test]
    fn sibling_groups_are_deduplicated_and_ordered() {
        // sysfs lists the same group once per sibling, so a raw read repeats it.
        let topology = CoreTopology::from_sibling_groups(vec![
            vec![2, 3],
            vec![2, 3],
            vec![0, 1],
            vec![0, 1],
            vec![],
        ]);
        assert_eq!(topology.cores(), &[vec![0, 1], vec![2, 3]]);
        assert_eq!(topology.core_count(), 2);
        assert_eq!(topology.logical_count(), 4);
        assert!(topology.has_smt());
    }

    #[test]
    fn overlapping_groups_never_double_count_a_cpu() {
        // A malformed/partial view must not produce a core count above the
        // number of logical CPUs, which would inflate the spin cap.
        let topology = CoreTopology::from_sibling_groups(vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
        assert_eq!(topology.logical_count(), 4);
        assert_eq!(topology.core_count(), 3);
        assert_eq!(topology.cores(), &[vec![0, 1], vec![2], vec![3]]);
    }

    #[test]
    fn compact_mask_on_adjacent_smt_covers_half_the_cores() {
        // The §26.2(b) failure: `DECODE_THREADS=16` confines to CPUs 0-15, which
        // is 8 physical cores, and 15 spinning workers were placed on them.
        let topology = adjacent_smt(16);
        let compact: Vec<usize> = (0..16).collect();
        assert_eq!(topology.physical_cores_within(&compact), 8);
        assert_eq!(
            topology.leaders_within(&compact),
            vec![0, 2, 4, 6, 8, 10, 12, 14]
        );
    }

    #[test]
    fn split_enumeration_compact_mask_covers_every_core() {
        // Same mask, different sibling enumeration: CPUs 0-15 on a `cpu`/`cpu+16`
        // host are 16 *distinct* cores, so the cap must not fire. Hardcoding
        // "half the mask" would be wrong here, which is why siblings are read
        // rather than assumed.
        let topology = split_smt(16);
        let compact: Vec<usize> = (0..16).collect();
        assert_eq!(topology.physical_cores_within(&compact), 16);
        assert_eq!(topology.leaders_within(&compact), compact);
    }

    #[test]
    fn leaders_stay_inside_the_allowed_mask() {
        // Pinning must never widen the process's CPU set: every leader chosen
        // for an odd-CPU mask is odd.
        let topology = adjacent_smt(8);
        let odd: Vec<usize> = (0..16).filter(|c| c % 2 == 1).collect();
        let leaders = topology.leaders_within(&odd);
        assert_eq!(leaders, odd);
        assert!(leaders.iter().all(|cpu| odd.contains(cpu)));
    }

    #[test]
    fn unknown_cpus_count_as_their_own_core() {
        // An incomplete topology must over-count, never collapse the cap.
        let topology = CoreTopology::from_sibling_groups(vec![vec![0, 1]]);
        assert_eq!(topology.physical_cores_within(&[0, 1, 7, 9]), 3);
        assert_eq!(topology.leaders_within(&[0, 1, 7, 9]), vec![0, 7, 9]);
    }

    #[test]
    fn no_smt_host_is_the_identity() {
        let topology = no_smt(12);
        let all: Vec<usize> = (0..12).collect();
        assert!(!topology.has_smt());
        assert_eq!(topology.physical_cores_within(&all), 12);
        assert_eq!(topology.leaders_within(&all), all);
    }

    #[test]
    fn empty_allowed_set_reports_no_cores() {
        assert_eq!(adjacent_smt(4).physical_cores_within(&[]), 0);
        assert!(adjacent_smt(4).leaders_within(&[]).is_empty());
    }

    #[test]
    fn siblings_of_finds_the_group() {
        let topology = adjacent_smt(4);
        assert_eq!(topology.siblings_of(5), Some(&[4, 5][..]));
        assert_eq!(topology.siblings_of(99), None);
    }

    #[test]
    fn cpu_list_parser_handles_both_sysfs_spellings() {
        assert_eq!(parse_cpu_list("0-1\n"), vec![0, 1]);
        assert_eq!(parse_cpu_list("0,16\n"), vec![0, 16]);
        assert_eq!(parse_cpu_list(" 2 - 3 , 8 "), vec![2, 3, 8]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
        // A reversed range is malformed; it must yield nothing rather than
        // panicking or looping.
        assert_eq!(parse_cpu_list("5-2"), Vec::<usize>::new());
        // An absurdly wide range is malformed too, and this one is expensive
        // rather than merely wrong: expanding it would allocate. Drop it, and
        // keep the well-formed siblings either side of it.
        assert_eq!(parse_cpu_list("0-4294967295"), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("0,1-4294967295,7"), vec![0, 7]);
        // The bound is on the span, not on the endpoints: a narrow range high
        // up is still parsed.
        assert_eq!(
            parse_cpu_list("1000000-1000001"),
            vec![1_000_000, 1_000_001]
        );
    }

    #[test]
    fn cap_never_returns_zero_for_a_real_request() {
        // Whatever this host reports, a request for one worker stays one.
        assert_eq!(cap_spinning_workers(1), 1);
        assert_eq!(cap_spinning_workers(0), 0);
    }

    #[test]
    fn cap_never_inflates_a_request() {
        assert!(cap_spinning_workers(2) <= 2);
        assert!(cap_spinning_workers(64) <= 64);
    }

    #[test]
    fn detected_host_topology_is_self_consistent() {
        // Runs on whatever CI machine this lands on; asserts the invariants
        // rather than a specific layout, so it is not host-fitted.
        //
        // Fails closed: this used to `return` on `None`, which meant a
        // detection regression made it pass vacuously. It is one of the three
        // tests the mutation battery caught doing exactly that.
        let topology = match require_host_for_placement() {
            Ok(topology) => topology,
            Err(reason) => {
                eprintln!("skipping {}: {reason}", module_path!());
                return;
            }
        };
        assert!(topology.core_count() > 0);
        assert!(topology.logical_count() >= topology.core_count());
        let mut seen = BTreeSet::new();
        for group in topology.cores() {
            assert!(!group.is_empty());
            assert!(
                group.windows(2).all(|w| w[0] < w[1]),
                "group not sorted: {group:?}"
            );
            for &cpu in group {
                assert!(seen.insert(cpu), "cpu {cpu} appears in two cores");
            }
        }
        let cores: Vec<Vec<usize>> = topology.cores().to_vec();
        assert!(
            cores.windows(2).all(|w| w[0][0] < w[1][0]),
            "cores not ordered by first cpu"
        );
    }

    /// Non-vacuous on Windows: `GetLogicalProcessorInformationEx` is always
    /// available, so detection must *succeed* here — a `None` means the
    /// flexible-array walk in `windows_cores::processor_cores` dropped every
    /// core (the way a wrong `GroupMask` offset does), which silently disables
    /// the SMT/physical-core cap and lets the task pool oversubscribe every
    /// logical CPU. The self-consistent test above returns early on `None`, so
    /// it cannot catch that; this one asserts the precondition held.
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_detection_succeeds_and_covers_every_logical_cpu() {
        let topology = host().expect("Windows core-topology detection returned None");
        assert!(topology.core_count() > 0, "no physical cores discovered");
        // `available_parallelism` counts the logical CPUs the process may use;
        // the discovered topology must cover at least that many, or the cap is
        // being computed against a truncated view of the machine.
        let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert!(
            topology.logical_count() >= logical,
            "topology covers {} logical CPUs but the process sees {logical}",
            topology.logical_count()
        );
        // On an SMT or hybrid part the cap must actually reduce below the
        // logical count; on a non-SMT part it is the identity. Either way it is
        // never zero for a non-zero request and never exceeds the physical
        // cores the process may run on.
        let capped = cap_spinning_workers(logical);
        assert!(capped >= 1 && capped <= logical);
        assert!(
            capped <= topology.core_count(),
            "cap {capped} exceeds {} physical cores",
            topology.core_count()
        );
    }

    /// Non-vacuous on Linux, for the same reason as the Windows test above and
    /// with more riding on it, because Linux is where the decode pool is
    /// actually exercised in CI.
    ///
    /// Every placement assertion this crate added in #1805 reaches its subject
    /// through `host()` returning `Some`: `planned_placement_is_one_worker_per_physical_core`
    /// opens with `host()?`, so it answers `None` when the topology is
    /// undiscoverable, and each caller then skips. That includes the
    /// `saw_placement_check` anti-vacuity guard in the width sweep, which is
    /// itself written `if allowed_now >= 2 && core_topology::host().is_some()`.
    /// So a detection regression to `None` would not fail anything -- it would
    /// quietly convert the placement half of the decode-pool contract into a
    /// no-op *and* silence the guard whose whole job is to say that happened.
    ///
    /// That is the #1792 shape one level down. #1792 was a pool reporting
    /// `realized=16 as_requested` while running on half the cores it claimed;
    /// this would be the check for it reporting nothing at all, which is
    /// indistinguishable from the check passing. A label nothing can check is
    /// a label that drifts, and a checker nothing can check is worse.
    ///
    /// This cannot be flaky on a real kernel. After the `to_str` fix in this
    /// change, `detect_linux` returns `None` in exactly two cases: the sysfs
    /// directory cannot be read at all, or it contains no `cpuN` entries. A CPU
    /// exposing neither `core_cpus_list` nor `thread_siblings_list` still falls
    /// back to being its own core, and entry names are kernel-generated ASCII,
    /// so any host that can run this crate's decode pool yields a non-empty
    /// topology. A Linux host with neither a readable `/sys` nor a `cpu0`
    /// cannot run the pool this asserts about either.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_detection_succeeds_so_placement_checks_cannot_silently_skip() {
        let topology = host().expect(
            "Linux core-topology detection returned None, which silently turns every \
             placement assertion in this crate into a no-op, including the anti-vacuity \
             guard that exists to report exactly that",
        );
        assert!(topology.core_count() > 0, "no physical cores discovered");
        // The discovered topology enumerates the machine, while
        // `available_parallelism` reports the CPUs this process may use, so the
        // former must cover the latter. A shortfall means the sysfs walk
        // dropped CPUs, which understates the core count and lets the cap
        // oversubscribe.
        let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert!(
            topology.logical_count() >= logical,
            "topology covers {} logical CPUs but the process sees {logical}",
            topology.logical_count()
        );
    }

    /// The anti-vacuity guard for the *other two* capabilities.
    ///
    /// `linux_detection_succeeds_so_placement_checks_cannot_silently_skip`
    /// proves detection answers. The placement falsifiers need three things,
    /// not one: detection, at least two physical cores, and an environment that
    /// accepts `sched_setaffinity`. The existing guard is satisfied on hosts
    /// where the other two are absent -- a 2-vCPU runner exposing one SMT pair
    /// reports `core_count() == 1`, and every placement check returns early
    /// while the guard reports the machine is fine.
    ///
    /// So this is deliberately *not* an unconditional assertion. A developer
    /// laptop in a 1-core container is a legitimate environment and reddening
    /// it gets the guard deleted. `NXRT_REQUIRE_PLACEMENT_TESTS=1` is the lane
    /// saying "here, absence is a defect" -- the same shape as
    /// `NXRT_REQUIRE_ORT_TESTS=1` on the `cli-ort` job.
    #[test]
    fn placement_capabilities_are_present_when_the_lane_requires_them() {
        if !placement_tests_required() {
            eprintln!(
                "{REQUIRE_PLACEMENT_ENV} is unset, so placement capabilities are not required \
                 here; the placement checks in this crate may be skipping"
            );
            return;
        }
        assert!(
            crate::decode_affinity::affinity_observation_supported(),
            "{REQUIRE_PLACEMENT_ENV}=1 on a target with no affinity backend. Placement is \
             unanswerable there by construction, so requiring it can only ever be a false \
             failure -- unset it for this lane."
        );
        let topology = require_host_for_placement()
            .expect("affinity is supported here, so detection must be too");
        assert!(
            topology.core_count() >= 2,
            "{REQUIRE_PLACEMENT_ENV}=1 but this host reports {} physical core(s) across {} \
             logical CPUs. Every `one worker per physical core` check in this crate returns \
             early below two, so this lane is running them as no-ops.",
            topology.core_count(),
            topology.logical_count()
        );
        // Probing the CPUs the placement checks actually pin to, not a
        // representative one: a confined process (cgroup cpuset, `taskset`)
        // can pin to some CPUs and not others, and the checks derive their
        // targets from the machine's topology rather than from the allowed set.
        let cpus: Vec<usize> = (0..topology.logical_count()).collect();
        let pinnable = crate::decode_affinity::order_pin_targets(&cpus, Some(topology))
            .into_iter()
            .take(2)
            .all(|cpu| {
                std::thread::spawn(move || {
                    crate::decode_affinity::pin_current_thread_to_cpu(cpu).is_ok()
                })
                .join()
                .unwrap_or(false)
            });
        assert!(
            pinnable,
            "{REQUIRE_PLACEMENT_ENV}=1 but this environment refuses to pin a thread to the first \
             two CPUs the placement policy would choose out of {} logical CPUs, so every \
             observed-placement check skips here.",
            topology.logical_count()
        );
    }

    /// The binding case, forced rather than waited for.
    ///
    /// `placement_capabilities_are_present_when_the_lane_requires_them` above
    /// asks what the *host* has. A run bound to one SMT sibling per core --
    /// `taskset -c 16,18,20,22`, which is how a careful measurement binds --
    /// satisfies every one of those assertions and still leaves the actual-mask
    /// control unable to build a shared-core layout. Measured on this repo's
    /// host before the gate existed: that band ran the control green with
    /// `NXRT_REQUIRE_PLACEMENT_TESTS=1` set, printing only a skip line that a
    /// captured CI log never shows.
    ///
    /// Both states of `required` and both causes of a missing layout are
    /// asserted here, because the exemption is the part that decays: if a
    /// genuine no-SMT host panicked, the requirement would be deleted rather
    /// than the lane fixed.
    #[test]
    fn a_layout_missing_only_because_of_binding_fails_closed_when_required() {
        let smt = CoreTopology::from_sibling_groups([vec![0, 1], vec![2, 3]]);
        let sibling_avoiding: BTreeSet<usize> = [0, 2].into_iter().collect();

        let forced = std::panic::catch_unwind(|| {
            placement_cpus_or_fail_closed(
                PlacementLayout::SharedCore,
                &smt,
                &sibling_avoiding,
                true,
                "a placement control",
            )
        });
        assert!(
            forced.is_err(),
            "an SMT host bound to one sibling per core reported success for a control that \
             cannot construct a shared core, in a lane that declared placement mandatory"
        );

        assert_eq!(
            placement_cpus_or_fail_closed(
                PlacementLayout::SharedCore,
                &smt,
                &sibling_avoiding,
                false,
                "a placement control",
            ),
            None,
            "a developer who has not declared placement mandatory must get a stated skip, not a \
             panic"
        );

        let no_smt = CoreTopology::from_sibling_groups([vec![0], vec![1]]);
        let everything: BTreeSet<usize> = [0, 1].into_iter().collect();
        assert_eq!(
            placement_cpus_or_fail_closed(
                PlacementLayout::SharedCore,
                &no_smt,
                &everything,
                true,
                "a placement control",
            ),
            None,
            "a host with no SMT anywhere cannot express a shared core, so requiring the control \
             there would be a false failure -- the exemption this gate depends on"
        );

        assert_eq!(
            placement_cpus_or_fail_closed(
                PlacementLayout::SharedCore,
                &smt,
                &[0, 1].into_iter().collect(),
                true,
                "a placement control",
            ),
            Some(vec![0, 1]),
            "the layout is representable inside the allowed set, so the gate must hand back the \
             CPUs rather than skip"
        );
    }

    /// The positive arm has the same shape, and a `DistinctCores` gate that
    /// silently answered `None` would remove the arm that stops the negative
    /// control being satisfied by an always-false predicate.
    #[test]
    fn a_single_allowed_core_fails_closed_for_the_distinct_core_layout() {
        let smt = CoreTopology::from_sibling_groups([vec![0, 1], vec![2, 3]]);
        let one_core: BTreeSet<usize> = [0, 1].into_iter().collect();

        let forced = std::panic::catch_unwind(|| {
            placement_cpus_or_fail_closed(
                PlacementLayout::DistinctCores,
                &smt,
                &one_core,
                true,
                "a placement control",
            )
        });
        assert!(
            forced.is_err(),
            "a two-core host bound to a single core reported success for a control that needs \
             two distinct cores, in a lane that declared placement mandatory"
        );

        let single = CoreTopology::from_sibling_groups([vec![0, 1]]);
        assert_eq!(
            placement_cpus_or_fail_closed(
                PlacementLayout::DistinctCores,
                &single,
                &one_core,
                true,
                "a placement control",
            ),
            None,
            "a one-core host cannot express two distinct cores at all; that is a platform fact \
             and `require_two_cores_for_placement` is what reports it"
        );

        assert_eq!(
            placement_cpus_or_fail_closed(
                PlacementLayout::DistinctCores,
                &smt,
                &[0, 2].into_iter().collect(),
                true,
                "a placement control",
            ),
            Some(vec![0, 2]),
            "one CPU from each of two cores is exactly the layout this arm needs"
        );
    }

    /// The mutation test, made permanent.
    ///
    /// #1805's placement checks were all reached through `host()` returning
    /// `Some`, so forcing detection to `None` made three of them -- including
    /// `the_planner_reports_a_shared_core_as_a_defect`, the *negative control* --
    /// pass vacuously. That was proved once by hand-editing `detect_linux`,
    /// which is exactly the kind of evidence that decays: the next person to
    /// reintroduce a silent skip has nothing to trip over.
    ///
    /// This forces the failure through the same policy function every placement
    /// test now routes through, so the proof runs on every CI job. It injects
    /// `None` as an argument rather than mutating a global, because the test
    /// binary is multi-threaded and a global toggle would make *other* tests
    /// observe a missing topology at random.
    #[test]
    fn forcing_a_detection_failure_fails_closed_where_detection_is_supported() {
        let forced = std::panic::catch_unwind(|| topology_or_fail_closed(None, true));
        assert!(
            forced.is_err(),
            "a forced detection failure on a supported target did not panic, so placement \
             assertions would skip silently instead of failing -- this is the #1805 \
             fail-open reintroduced"
        );

        let message = forced
            .err()
            .and_then(|payload| {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            })
            .unwrap_or_default();
        assert!(
            message.contains("placement"),
            "the fail-closed panic must say what it is about, so a CI log identifies the \
             defect without a bisect; got {message:?}"
        );
    }

    /// The other half: a skip is still allowed where there is genuinely no
    /// backend, and it has to carry a reason. Without this, "fail closed" would
    /// be indistinguishable from "panic on every platform we have not ported
    /// to", and the pressure would be to weaken the panic rather than state the
    /// exemption.
    #[test]
    fn an_unsupported_target_skips_with_an_explicit_stated_reason() {
        let skipped = topology_or_fail_closed(None, false)
            .expect_err("an unsupported target must skip, not resolve a topology");
        assert!(
            !skipped.trim().is_empty(),
            "a skip must state why; a bare skip is the silent-return this change removes"
        );
        assert_eq!(skipped, NO_BACKEND_REASON);
    }

    /// A present topology resolves on either setting -- so the two tests above
    /// are testing the `None` handling specifically, rather than a function
    /// that panics unconditionally.
    #[test]
    fn a_detected_topology_resolves_regardless_of_support_flag() {
        let detected = match require_host_for_placement() {
            Ok(detected) => detected,
            // Only reachable on a target with no backend, where the two tests
            // above already pin the behaviour.
            Err(reason) => {
                eprintln!("skipping the resolve-either-way check: {reason}");
                return;
            }
        };
        assert!(topology_or_fail_closed(Some(detected), true).is_ok());
        assert!(topology_or_fail_closed(Some(detected), false).is_ok());
    }

    /// This crate's own target must be one that fails closed. Without it,
    /// porting to a target that quietly lands outside `DETECTION_SUPPORTED`
    /// would re-open every placement skip with no test noticing.
    #[test]
    // The assertion is deliberately constant-valued: it is a *compile-target*
    // tripwire, not a runtime check. A `const` block would turn a port to an
    // unported target into a build error, which would break the graceful skip
    // this change is careful to preserve.
    #[allow(clippy::assertions_on_constants)]
    fn the_targets_this_crate_is_tested_on_are_all_fail_closed() {
        assert!(
            DETECTION_SUPPORTED,
            "this target has no core-topology backend, so every placement assertion in \
             this crate is skipping; if that is intended, say so here explicitly"
        );
        assert!(require_host_for_placement().is_ok());
    }
}
