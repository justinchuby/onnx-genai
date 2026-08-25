//! Reads the advisory host lock so a measured row can record whether anybody
//! declared the machine while it was being measured.
//!
//! # Why a reader, when `scripts/hostlock.sh` already exists
//!
//! The lock has been on `main` for a while: atomic `mkdir`, owner and anchor PID
//! in a metadata file, `/proc` liveness plus a TTL for staleness, and a `--gate`
//! on the instantaneous runnable count. It is a good lock. But
//! `grep -r hostlock crates/` returns **nothing** -- no benchmark, no harness and
//! no result row consumes it. A capability that exists, is `pub`, and has no
//! caller is indistinguishable in the output from one that was never built, and
//! the absence reads as success.
//!
//! Two agents sharing this host each ran a benchmark today while the other
//! believed the box was quiet, and each checked with `ps` first. Both checks
//! were honest and both were wrong, because a saturating test arm lives roughly
//! 30-40 seconds -- long enough to be seen, short enough to be gone before the
//! other party reacts. Etiquette cannot close that window. Recording what the
//! lock said, in the row, can at least stop the resulting number from being
//! believed later.
//!
//! # What this is not
//!
//! It does not acquire, release or enforce anything, and it must not: taking a
//! lock is a decision a harness makes, and a library that took one as a side
//! effect of formatting a field would be far worse than no lock at all. This
//! only reads.
//!
//! It also cannot tell you the host was quiet. The lock records *declared
//! intent*; [`Contention`](crate::Contention) measures what actually happened.
//! They answer different questions and a row wants both -- an unlocked run on a
//! genuinely idle box is fine, and a locked run next to somebody's unlocked
//! `cargo test` is not.
//!
//! # Both ends of the window, not one
//!
//! [`field`] takes two readings and reports [`Changed`](LockField::Changed) when
//! they disagree. This is the whole point of the module. A single reading at the
//! end of a run would have reported a clean, plausible holder for a window that
//! changed hands halfway through -- which is exactly the stale-snapshot error
//! that produced the contaminated measurements in the first place, just moved
//! from `ps` into the row where it would be harder to spot.

use std::fmt;

/// Where the lock lives. Matches `scripts/hostlock.sh`, which must stay the
/// single source of truth for the path: a reader that looked somewhere else
/// would report `free` forever and do so convincingly.
///
/// Public so the agreement test can assert it against the script's own
/// default rather than against a second copy of the string. A divergence here
/// is silent in the direction that permits a run, so it is asserted rather
/// than reviewed.
#[cfg(target_os = "linux")]
pub const DEFAULT_LOCK_DIR: &str = "/tmp/onnx-genai-hostlock";

/// Longest owner name that will be printed. Long enough for a name, short
/// enough that a pathological one cannot dominate a result row.
const MAX_OWNER_LEN: usize = 32;

/// Who claims the host, as recorded in the lock's metadata file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockHolder {
    /// Sanitised for printing; see `sanitise_owner`.
    pub owner: String,
    /// The owner exactly as written, minus surrounding whitespace.
    ///
    /// Attribution compares this and never [`owner`](Self::owner), because
    /// sanitising is lossy in the one direction that matters: it maps
    /// `sebastian!` and any 33-character name sharing a 32-character prefix
    /// onto an existing name, and a collision there turns `foreign` into
    /// `mine` and marks a contaminated row protected. Display may be lossy;
    /// the protection decision may not.
    pub owner_raw: String,
    /// The PID whose liveness decides whether the lock is stale. `None` when the
    /// metadata did not carry one, which is itself a reason to distrust it.
    pub anchor_pid: Option<u32>,
    /// Field 22 of the anchor's `/proc/<pid>/stat`, as recorded when the lock
    /// was taken. Without it a recycled PID reads as the original holder, so a
    /// lock whose owner exited half an hour ago can look current.
    pub start_time: Option<u64>,
    /// What the holder said they were running. Sanitised the same way.
    pub reason: String,
}

/// The lock's state at one instant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LockState {
    /// The lock directory could not be inspected -- not "absent", which is
    /// [`Free`](LockState::Free), but unreadable, or a platform with no
    /// `/proc` to check liveness against. Distinct from `Free` on purpose: "no
    /// one holds it" and "I could not tell" must never format the same.
    Unknown,
    /// Nobody holds it.
    Free,
    /// Nobody holds it and nobody *can*: the lock directory does not exist and
    /// cannot be created here, so on this host the declaration protocol is
    /// unavailable.
    ///
    /// Distinct from [`Free`](LockState::Free) for the same reason
    /// `hostlock.sh` grew its `UNUSABLE` state and exit 7: absent-because-idle
    /// and absent-because-impossible are the same bytes on disk and opposite
    /// facts about the host. Reporting the second as the first is a fail-open
    /// -- a reader on a misconfigured box would stamp `host_lock=free` on every
    /// row forever, and `free` reads as "nobody had declared the machine" when
    /// the truth is "nobody could have, including the peer whose benchmark was
    /// running alongside this one".
    Unusable,
    /// Held, and the anchor process is provably alive.
    Held(LockHolder),
    /// Held, and liveness could not be established -- no anchor PID, no
    /// recorded start time, or a `/proc` that would not answer. Distinct from
    /// both neighbours on purpose, and `hostlock.sh` makes the same
    /// distinction: "cannot verify liveness" must not be treated as "the holder
    /// is dead", or an unparseable lock caught mid-write gets somebody's
    /// machine taken out from under them. It is equally not proof of life, so
    /// it never certifies a row either.
    Unverified(LockHolder),
    /// Held by a process that is gone. The load that lock was covering may well
    /// still be running -- reaping a lock does not stop an orphaned benchmark --
    /// so this is a warning, not a synonym for free.
    Stale(LockHolder),
}

impl LockState {
    fn holder(&self) -> Option<&LockHolder> {
        match self {
            LockState::Held(holder) | LockState::Unverified(holder) | LockState::Stale(holder) => {
                Some(holder)
            }
            LockState::Unknown | LockState::Free | LockState::Unusable => None,
        }
    }
}

/// The value of the `host_lock=` field on a result row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LockField {
    /// Could not read the lock at either end.
    Unknown,
    /// Free for the whole window. Not an error: it means the row was taken
    /// without a declaration, which is worth knowing precisely because it is
    /// the state every unprotected run is in.
    Free,
    /// The host cannot take the lock at all, so this row could not have been
    /// protected and neither could anyone else's. Not `free`: the remedy is a
    /// `lock_dir=` that works, not a retry.
    Unusable,
    /// Held throughout by us -- the owner matched `HOSTLOCK_OWNER`.
    Mine(String),
    /// Held throughout by somebody else. Every number in this row was measured
    /// while another participant had declared the machine.
    Foreign(String),
    /// Held throughout, but `HOSTLOCK_OWNER` was not set, so the reader cannot
    /// say whether that was us. Reported rather than guessed.
    Held(String),
    /// Held throughout by a dead anchor process.
    Stale(String),
    /// Held throughout, but liveness could never be established. Neither a
    /// protected row nor a free host.
    Unverified(String),
    /// The two readings disagree. The row spans a change of custody and no
    /// single holder describes it.
    Changed,
}

impl fmt::Display for LockField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockField::Unknown => write!(f, "unknown"),
            LockField::Free => write!(f, "free"),
            LockField::Unusable => write!(f, "unusable"),
            LockField::Mine(owner) => write!(f, "mine:{owner}"),
            LockField::Foreign(owner) => write!(f, "foreign:{owner}"),
            LockField::Held(owner) => write!(f, "held:{owner}"),
            LockField::Stale(owner) => write!(f, "stale:{owner}"),
            LockField::Unverified(owner) => write!(f, "unverified:{owner}"),
            LockField::Changed => write!(f, "changed"),
        }
    }
}

impl LockField {
    /// Whether this row was taken under a declaration that covered the whole
    /// window and belonged to the process making the measurement.
    ///
    /// [`Held`](LockField::Held) is deliberately **not** protected: without
    /// `HOSTLOCK_OWNER` the reader cannot tell our own lock from somebody
    /// else's, and resolving that ambiguity in the flattering direction is how a
    /// contaminated row acquires a clean label.
    pub fn is_protected(&self) -> bool {
        matches!(self, LockField::Mine(_))
    }
}

/// Strips anything that could forge a field boundary, and truncates.
///
/// `hostlock.sh`'s own header documents this hazard against itself: an owner of
/// `gaff hostlock_state=FREE declared=no` splices two extra key/value pairs into
/// its one-line provenance output, and a consumer reading the *first*
/// `hostlock_state=` gets `FREE` for a held lock. The same string would do the
/// same thing to a result row. Whitespace and `=` are therefore replaced rather
/// than trusted, and the metadata file is attacker-adjacent in the only sense
/// that matters here -- any participant on the box can write any owner they
/// like, including by accident.
///
/// An empty or entirely-unprintable owner becomes `?`, so the field is never
/// empty; `host_lock=held:` reads as a parse failure in the consumer rather than
/// as a nameless holder.
fn sanitise_owner(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(MAX_OWNER_LEN)
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "?".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Pulls one `key=value` out of the lock's metadata file.
///
/// First match wins, matching `hostlock.sh`'s own `sed -n 's/^key=//p' | head
/// -1`. Agreeing with the writer matters more than any better rule: a reader
/// that took the *last* duplicate would disagree with the tool about who holds
/// the lock, and only under exactly the corrupted metadata where being right
/// counts.
fn meta_get<'a>(meta: &'a str, key: &str) -> Option<&'a str> {
    meta.lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
}

/// Parses metadata into a holder. `None` when there is no `owner` line at all,
/// which is how a half-written file is distinguished from a held lock.
pub fn parse_meta(meta: &str) -> Option<LockHolder> {
    let owner = meta_get(meta, "owner")?;
    Some(LockHolder {
        owner: sanitise_owner(owner),
        owner_raw: owner.trim().to_string(),
        // A non-numeric or absent anchor is `None` rather than a default. There
        // is no safe default: 0 or 1 would both name a live process and label a
        // dead holder's lock as current.
        anchor_pid: meta_get(meta, "anchor_pid").and_then(|pid| pid.trim().parse::<u32>().ok()),
        start_time: meta_get(meta, "start_time").and_then(|t| t.trim().parse::<u64>().ok()),
        reason: sanitise_owner(meta_get(meta, "reason").unwrap_or("")),
    })
}

/// The three facts about a process that decide whether a lock still means
/// anything, read together from one `/proc` snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcInfo {
    /// Field 3 of `/proc/<pid>/stat`.
    pub state: char,
    /// Field 22 of `/proc/<pid>/stat`. Distinguishes the original process from
    /// a later one that happens to have been given the same PID.
    pub start_time: u64,
    /// `Threads:` from `/proc/<pid>/status`, when it could be read.
    pub threads: Option<u64>,
}

/// What can be established about the anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Liveness {
    Alive,
    Dead,
    /// Not provable either way. Never collapsed into one of the other two:
    /// calling it dead invites a takeover of a machine somebody is using, and
    /// calling it alive certifies a row on evidence that does not exist.
    Unprovable,
}

/// Decides liveness from the recorded anchor and a `/proc` reading.
///
/// This mirrors `anchor_alive` and `pid_is_live` in `scripts/hostlock.sh`
/// deliberately and in detail. The script's own comment on that function says
/// the last time two call sites each decided this question for themselves the
/// answers disagreed, and two of the four defects in #1830 came out of the gap.
/// A Rust reader that decided it independently would be a third call site, and
/// it would disagree in the worst possible place: the row that gets published.
///
/// The two cases a naive `/proc/<pid>` existence test gets wrong, both of which
/// this reader got wrong before the agreement test caught them:
///
/// * a **zombie** still has a `/proc/<pid>` entry and its start time still
///   matches, so a `Path::exists` check reports a held lock on a corpse forever.
///   Every agent harness here launches long commands without an immediate
///   `wait()`, so this is the common shape, not the exotic one;
/// * a **recycled PID** passes an existence test as well, which is what the
///   recorded start time is for.
///
/// The zombie rule is not simply "state Z means dead". When a thread-group
/// leader exits via `pthread_exit` while its other threads keep running,
/// `/proc/<tgid>/stat` reports `Z` for a fully live process. `Threads:` is the
/// count of non-reaped tasks in the group, so a true zombie reads 1 and a live
/// leader reads more; when it cannot be read there is no evidence of death.
pub fn liveness(holder: &LockHolder, info: Option<ProcInfo>) -> Liveness {
    let Some(pid) = holder.anchor_pid else {
        return Liveness::Unprovable;
    };
    let _ = pid;
    let Some(info) = info else {
        // No `/proc` entry at all is the one unambiguous death.
        return Liveness::Dead;
    };
    if info.state == 'Z' && info.threads.is_some_and(|t| t <= 1) {
        return Liveness::Dead;
    }
    match holder.start_time {
        Some(recorded) if recorded != info.start_time => Liveness::Dead,
        Some(_) => Liveness::Alive,
        // A running anchor with no recorded start time is exactly the script's
        // `unverifiable_live_anchor`: not dead, and not verified either.
        None => Liveness::Unprovable,
    }
}

/// Reads the three fields [`liveness`] needs, or `None` when the process is
/// gone.
#[cfg(target_os = "linux")]
pub fn proc_info(pid: u32) -> Option<ProcInfo> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The comm field is parenthesised and may itself contain spaces and
    // parentheses, so the split is on the LAST `)`, exactly as the script's
    // `sed 's/.*) //'` does. Splitting on whitespace instead would misfield
    // every process whose name contains a space.
    let rest = &stat[stat.rfind(')')? + 1..];
    let mut fields = rest.split_whitespace();
    let state = fields.next()?.chars().next()?;
    // After the comm field, state is field 1 and starttime is field 20.
    let start_time = fields.nth(18)?.parse::<u64>().ok()?;
    let threads = std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|l| l.strip_prefix("Threads:"))
                .and_then(|t| t.trim().parse::<u64>().ok())
        });
    Some(ProcInfo {
        state,
        start_time,
        threads,
    })
}

/// Classifies already-read metadata. Split from the filesystem so the decision
/// table can be tested without a lock on the box, and so a test can never leave
/// one behind.
pub fn classify(meta: Option<&str>, probe: impl Fn(u32) -> Option<ProcInfo>) -> LockState {
    let Some(meta) = meta else {
        return LockState::Free;
    };
    let Some(holder) = parse_meta(meta) else {
        return LockState::Unknown;
    };
    let info = holder.anchor_pid.and_then(probe);
    match liveness(&holder, info) {
        Liveness::Alive => LockState::Held(holder),
        Liveness::Dead => LockState::Stale(holder),
        Liveness::Unprovable => LockState::Unverified(holder),
    }
}

/// Reads the lock as it stands right now.
///
/// Resolves the directory exactly as `hostlock.sh` does -- `HOSTLOCK_DIR`, then
/// the machine-local config's `lock_dir=`, then the default. The two
/// implementations resolving it differently is not a cosmetic divergence: on a
/// host whose config has moved the lock, a reader still looking at `/tmp` finds
/// nothing and stamps `host_lock=free` on every row of a run that was taken
/// while a peer held the box. That is the mislabelling this module exists to
/// prevent, reintroduced one layer down, so the rule lives in
/// [`resolve_lock_dir`] and both sides are held to it by
/// `tests/agrees_with_hostlock_sh.rs`.
#[cfg(target_os = "linux")]
pub fn read() -> LockState {
    let dir = match resolve_lock_dir() {
        Ok(dir) => dir,
        // A config we cannot honour is not evidence that nobody holds the lock.
        // `hostlock.sh` refuses to run at all in this state; a reader has no
        // such option, so it says it does not know.
        Err(_) => return LockState::Unknown,
    };
    state_at(&dir)
}

/// [`read`], for an already-resolved directory.
///
/// Split out so the ordering below is reachable from the differential test
/// without setting `HOSTLOCK_DIR` on a shared process environment. The ordering
/// is the load-bearing part and it mirrors `lock_state` in the script exactly:
///
/// 1. **A lock directory that exists wins.** Whether the box is *also*
///    misconfigured is irrelevant once somebody has published a claim: the
///    honest answer for a second participant is "held", not "your host is
///    broken". A store that reported `unusable` here would relabel real
///    contention as a config fault, which is how a peer's live benchmark gets
///    walked over by someone off fixing their own machine.
/// 2. Only when it is absent does the reason for the absence matter, and that
///    is what [`dir_problem`] answers.
#[cfg(target_os = "linux")]
pub fn state_at(dir: &std::path::Path) -> LockState {
    if !dir.is_dir() {
        return match dir_problem(dir) {
            Some(_) => LockState::Unusable,
            None => LockState::Free,
        };
    }
    classify_io(
        std::fs::read_to_string(dir.join("meta")).map_err(|err| err.kind()),
        proc_info,
    )
}

/// Why a lock could not be created at `dir`, or `None` if one could.
///
/// The port of `lock_dir_problem` from the script, and it answers the same
/// question the same way: not "is the lock directory writable" -- it does not
/// exist yet -- but "can entries be created *beside* it". The script publishes
/// by staging at a sibling and `mv -T`-ing it into place, so the permission
/// that matters belongs to the nearest **existing ancestor**, which is where
/// `mkdir -p` would start building.
///
/// Both bits are required and both have bitten this design once. A directory
/// that is writable but not searchable (mode 0600) cannot hold entries at all
/// -- `mkdir 0600-parent/sub` fails `EACCES` -- so a `-w`-only check passes a
/// host on which nothing can be created.
///
/// `access(2)` is used rather than mode bits because the script's `test -w`
/// resolves to `faccessat`: it accounts for uid, supplementary groups, ACLs and
/// read-only mounts, and reimplementing that from `st_mode` would disagree with
/// the writer on exactly the hosts where the answer is interesting. Note the
/// shared blind spot, worth stating rather than discovering: `access` reports
/// what the *kernel* permits, so a directory the caller may write but is
/// forbidden to write by policy outside the filesystem reads as usable here and
/// in the script alike.
#[cfg(target_os = "linux")]
pub fn dir_problem(dir: &std::path::Path) -> Option<String> {
    if dir.exists() && !dir.is_dir() {
        return Some(format!("{} exists and is not a directory", dir.display()));
    }
    let mut probe = dir.parent().unwrap_or_else(|| std::path::Path::new("."));
    while !probe.exists() {
        probe = match probe.parent() {
            Some(parent) => parent,
            None => std::path::Path::new("."),
        };
    }
    if !probe.is_dir() {
        return Some(format!("{} exists and is not a directory", probe.display()));
    }
    if !can_create_in(probe) {
        return Some(format!(
            "{} is not writable by uid {}",
            probe.display(),
            // SAFETY: `getuid` reads a field of the calling process's
            // credentials. It cannot fail and takes no pointer.
            unsafe { libc::getuid() }
        ));
    }
    None
}

/// `test -w "$p" && test -x "$p"`, via the same syscall the shell uses.
#[cfg(target_os = "linux")]
fn can_create_in(dir: &std::path::Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c_path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        // An interior NUL cannot name a path any syscall will accept, so
        // nothing can be created there. Reporting it as usable would defer the
        // failure to `mkdir` and blame the wrong thing.
        return false;
    };
    // SAFETY: `c_path` is a NUL-terminated C string that outlives the call, and
    // `access` only reads it.
    unsafe { libc::access(c_path.as_ptr(), libc::W_OK | libc::X_OK) == 0 }
}

/// Nothing to read: liveness here depends on `/proc`, and there is no
/// equivalent to fall back on.
///
/// `proc_info` deliberately **does not exist** off Linux rather than returning
/// `None`, which [`liveness`] would read as an unambiguous death and report
/// every live lock as `stale` -- the reaper-shaped error, and one that would
/// have compiled silently. A caller that needs it on another platform gets a
/// compile error instead of a plausible wrong answer.
///
/// It is spelled without an intra-doc link on purpose: a link would name a
/// symbol that, by the whole point of this change, is absent here.
#[cfg(not(target_os = "linux"))]
pub fn read() -> LockState {
    LockState::Unknown
}

/// Why a lock directory could not be resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockDirError {
    /// The config file that carries the unusable value.
    pub config: std::path::PathBuf,
    /// The value as written, so the message can name it.
    pub value: String,
}

impl fmt::Display for LockDirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: lock_dir must be a non-empty absolute path (got '{}')",
            self.config.display(),
            self.value
        )
    }
}

/// The machine-local config file both implementations read.
#[cfg(target_os = "linux")]
pub fn config_path() -> std::path::PathBuf {
    if let Ok(explicit) = std::env::var("HOSTLOCK_CONF")
        && !explicit.is_empty()
    {
        return std::path::PathBuf::from(explicit);
    }
    let base = match std::env::var("XDG_CONFIG_HOME") {
        Ok(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
        _ => std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_else(|_| "/nonexistent".into()),
        )
        .join(".config"),
    };
    base.join("onnx-genai").join("hostlock.conf")
}

/// The `lock_dir=` value in a config file's text, before validation.
///
/// Pure, so the parsing rules -- first key wins, `#` starts a comment,
/// whitespace around both sides is not part of the path -- are testable without
/// a filesystem, and so they can be stated once for comparison against the
/// `sed` expression in the script.
pub fn parse_conf_lock_dir(text: &str) -> Option<String> {
    // ASCII whitespace only, deliberately, because the other implementation is
    // `sed`'s `[[:space:]]`, which is ASCII even in a UTF-8 locale. Rust's
    // `trim` follows Unicode `White_Space`, so a NO-BREAK SPACE beside the path
    // would be stripped here and kept there -- and the two implementations
    // would then resolve DIFFERENT directories from one config file, which is
    // the state in which the reader stamps `free` on rows taken while a peer
    // holds the box. An unlikely byte is not a reason to let the rule differ.
    let ascii_ws = |c: char| c.is_ascii_whitespace();
    for line in text.lines() {
        let line = line.trim_start_matches(ascii_ws);
        let Some(rest) = line.strip_prefix("lock_dir") else {
            continue;
        };
        let rest = rest.trim_start_matches(ascii_ws);
        let Some(value) = rest.strip_prefix('=') else {
            continue;
        };
        let value = value.split('#').next().unwrap_or("").trim_matches(ascii_ws);
        return Some(value.to_string());
    }
    None
}

/// `HOSTLOCK_DIR`, else the config's `lock_dir=`, else the default.
///
/// The env override is deliberately *not* equivalent to the config one: it is
/// per process, so it yields a lock that coordinates with nobody. The script
/// announces that on stderr; a library has nowhere to announce it, which is
/// why `provenance`/`status --porcelain` carry `lock_dir=` into the row
/// instead.
#[cfg(target_os = "linux")]
pub fn resolve_lock_dir() -> Result<std::path::PathBuf, LockDirError> {
    let env_dir = std::env::var("HOSTLOCK_DIR").ok().filter(|d| !d.is_empty());
    resolve_lock_dir_from(env_dir.as_deref(), &config_path())
}

/// [`resolve_lock_dir`], with its two inputs passed in.
///
/// The environment is process-global, so a test that set `HOSTLOCK_DIR` to
/// exercise the real entry point would leak the override into every other test
/// in the binary on any failing assertion. Taking the inputs as arguments is
/// what lets the differential test drive the *resolution* -- precedence,
/// absoluteness, the default fallback -- rather than only the parser, which
/// would leave the rule that actually picks the directory unexercised while
/// looking thoroughly tested.
#[cfg(target_os = "linux")]
pub fn resolve_lock_dir_from(
    env_dir: Option<&str>,
    config: &std::path::Path,
) -> Result<std::path::PathBuf, LockDirError> {
    if let Some(dir) = env_dir
        && !dir.is_empty()
    {
        return Ok(std::path::PathBuf::from(dir));
    }
    let Ok(text) = std::fs::read_to_string(config) else {
        return Ok(std::path::PathBuf::from(DEFAULT_LOCK_DIR));
    };
    match parse_conf_lock_dir(&text) {
        Some(value) if value.starts_with('/') => Ok(std::path::PathBuf::from(value)),
        // Present but unusable. Falling back to the default here would put half
        // the box on one path and half on the other, which is worse than either
        // choice: the admin who wrote the file may have done so precisely
        // because the default does not work on this host.
        Some(value) => Err(LockDirError {
            config: config.to_path_buf(),
            value,
        }),
        None => Ok(std::path::PathBuf::from(DEFAULT_LOCK_DIR)),
    }
}

/// Turns the result of reading the metadata file into a state.
///
/// Separated from [`read`] only so the error arms are reachable from a test.
/// They are the arms most likely to be got wrong and least likely to be
/// exercised: an unreadable lock directory has to stay distinguishable from an
/// absent one, because "nobody holds it" and "I could not tell" are different
/// claims and only one of them permits a run.
pub fn classify_io(
    meta: Result<String, std::io::ErrorKind>,
    probe: impl Fn(u32) -> Option<ProcInfo>,
) -> LockState {
    match meta {
        Ok(meta) => classify(Some(&meta), probe),
        // Absent is a measurement: nobody has taken the lock. Any other error is
        // not, and must not be laundered into `Free`.
        Err(std::io::ErrorKind::NotFound) => LockState::Free,
        Err(_) => LockState::Unknown,
    }
}

/// Reduces two readings, taken at the two ends of a measured window, to the
/// `host_lock=` field.
///
/// `self_owner` is `HOSTLOCK_OWNER` if the caller set it. Without it a held lock
/// can only be reported as [`Held`](LockField::Held): a reader cannot tell its
/// own declaration from a co-tenant's, and guessing would put a reassuring label
/// on precisely the rows that need a suspicious one.
pub fn field(before: &LockState, after: &LockState, self_owner: Option<&str>) -> LockField {
    if before != after {
        return LockField::Changed;
    }
    match before {
        LockState::Unknown => LockField::Unknown,
        LockState::Free => LockField::Free,
        LockState::Unusable => LockField::Unusable,
        LockState::Stale(holder) => LockField::Stale(holder.owner.clone()),
        LockState::Unverified(holder) => LockField::Unverified(holder.owner.clone()),
        LockState::Held(holder) => match self_owner.map(str::trim) {
            // An empty name on either side is the absence of an attribution,
            // not an attribution to nobody. Without this guard a holder whose
            // owner is blank and a `HOSTLOCK_OWNER` that is blank compare equal
            // and certify the row -- two unnamed parties matching each other.
            Some(mine) if mine.is_empty() || holder.owner_raw.is_empty() => {
                LockField::Foreign(holder.owner.clone())
            }
            Some(mine) if mine == holder.owner_raw => LockField::Mine(holder.owner.clone()),
            Some(_) => LockField::Foreign(holder.owner.clone()),
            None => LockField::Held(holder.owner.clone()),
        },
    }
}

/// [`field`], with `HOSTLOCK_OWNER` read from the environment.
///
/// There is deliberately **no fallback to `$USER`**, even though
/// `hostlock.sh`'s `--owner` has one. Every agent on this host runs as the same
/// user, so `$USER` cannot distinguish one declaration from another, and
/// defaulting to it would report `mine:` for a co-tenant's lock -- the one
/// direction this module exists to prevent.
///
/// `hostlock.sh run` exports the owner it was *given* -- `--owner`, or
/// `HOSTLOCK_OWNER` already in the environment -- so a wrapped command can
/// recognise the lock it is running under (#1929). It does **not** export an
/// owner it defaulted from `$USER`, for the reason above: the lock still
/// records that default, but nothing downstream is told to treat it as its
/// own. So a run that declared nothing reports [`Held`](LockField::Held),
/// which is unattributed and unprotected, and is the honest answer.
pub fn field_from_env(before: &LockState, after: &LockState) -> LockField {
    let owner = std::env::var("HOSTLOCK_OWNER").ok();
    field(before, after, owner.as_deref())
}

/// What the holder said they were running, when there is one and both readings
/// agree on it. For a human reading a row later, not for any decision.
pub fn reason(before: &LockState, after: &LockState) -> Option<String> {
    let (a, b) = (before.holder()?, after.holder()?);
    (a == b && !b.reason.is_empty() && b.reason != "?").then(|| b.reason.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(owner: &str, pid: &str) -> String {
        format!("anchor_pid={pid}\nstart_time=900\nowner={owner}\nreason=acc0\nttl=0\n")
    }

    fn running(start_time: u64) -> Option<ProcInfo> {
        Some(ProcInfo {
            state: 'S',
            start_time,
            threads: Some(4),
        })
    }

    fn held(owner: &str) -> LockState {
        LockState::Held(LockHolder {
            owner: owner.to_string(),
            owner_raw: owner.to_string(),
            anchor_pid: Some(1),
            start_time: Some(900),
            reason: "acc0".to_string(),
        })
    }

    /// The exact string `hostlock.sh`'s header names as its own provenance
    /// hazard. A row is a `key=value` line too, so it inherits the hazard.
    #[test]
    fn an_owner_cannot_splice_extra_fields_into_a_result_row() {
        let holder = parse_meta(&meta("gaff hostlock_state=FREE declared=no", "1"))
            .expect("owner line is present");
        assert!(
            !holder.owner.contains('=') && !holder.owner.contains(' '),
            "owner must not be able to forge a field boundary: {}",
            holder.owner
        );
        let rendered = field(
            &LockState::Held(holder.clone()),
            &LockState::Held(holder),
            None,
        )
        .to_string();
        assert_eq!(
            rendered.split_whitespace().count(),
            1,
            "the field must occupy exactly one token of the row: {rendered}"
        );
    }

    /// A newline is worse than a space: it would end the row entirely.
    #[test]
    fn a_newline_in_an_owner_cannot_terminate_the_row() {
        let holder = parse_meta("owner=roy\nanchor_pid=1\n").expect("owner line is present");
        assert_eq!(holder.owner, "roy");
        assert!(
            !field(
                &LockState::Held(holder.clone()),
                &LockState::Held(holder),
                None
            )
            .to_string()
            .contains('\n')
        );
    }

    /// Never empty: `host_lock=held:` reads as a broken parser, not as a lock
    /// held by nobody.
    #[test]
    fn an_unprintable_owner_is_named_rather_than_left_blank() {
        assert_eq!(sanitise_owner(""), "?");
        assert_eq!(sanitise_owner("   "), "?");
        assert_eq!(sanitise_owner("==="), "?");
    }

    #[test]
    fn a_long_owner_cannot_dominate_the_row() {
        assert_eq!(sanitise_owner(&"x".repeat(200)).len(), MAX_OWNER_LEN);
    }

    /// The reason this module reads twice. A single reading at the end would
    /// have called this window `mine:sebastian` and looked entirely credible.
    #[test]
    fn a_window_that_changed_hands_is_never_reported_as_one_holder() {
        assert_eq!(
            field(&held("roy"), &held("sebastian"), Some("sebastian")),
            LockField::Changed
        );
        assert_eq!(
            field(&LockState::Free, &held("sebastian"), Some("sebastian")),
            LockField::Changed
        );
        assert_eq!(
            field(&held("sebastian"), &LockState::Free, Some("sebastian")),
            LockField::Changed
        );
        assert!(
            !field(&LockState::Free, &held("sebastian"), Some("sebastian")).is_protected(),
            "a window that only became ours partway through was not protected"
        );
    }

    /// Held-by-us and held-by-someone-else are the two rows a reader most needs
    /// to tell apart, and they are the same `LockState`.
    #[test]
    fn the_owner_decides_whether_a_held_lock_protects_this_row() {
        assert_eq!(
            field(&held("sebastian"), &held("sebastian"), Some("sebastian")),
            LockField::Mine("sebastian".into())
        );
        assert_eq!(
            field(&held("roy"), &held("roy"), Some("sebastian")),
            LockField::Foreign("roy".into())
        );
        assert!(field(&held("sebastian"), &held("sebastian"), Some("sebastian")).is_protected());
        assert!(!field(&held("roy"), &held("roy"), Some("sebastian")).is_protected());
    }

    /// Without `HOSTLOCK_OWNER` the reader genuinely cannot attribute the lock,
    /// and must not resolve that in the flattering direction.
    #[test]
    fn an_unattributable_lock_is_reported_as_unprotected() {
        let f = field(&held("roy"), &held("roy"), None);
        assert_eq!(f, LockField::Held("roy".into()));
        assert!(
            !f.is_protected(),
            "an unattributed lock must never label a row as protected"
        );
    }

    /// `Free` and `Unknown` format differently on purpose: one is a
    /// measurement, the other is the absence of one.
    #[test]
    fn an_unreadable_lock_never_formats_as_a_free_one() {
        assert_eq!(
            field(&LockState::Free, &LockState::Free, None).to_string(),
            "free"
        );
        assert_eq!(
            field(&LockState::Unknown, &LockState::Unknown, None).to_string(),
            "unknown"
        );
        assert!(!field(&LockState::Free, &LockState::Free, Some("me")).is_protected());
    }

    /// A dead anchor does not mean a quiet host: reaping a lock does not stop
    /// the orphaned benchmark it was covering.
    #[test]
    fn a_departed_anchor_is_stale_rather_than_free() {
        let state = classify(Some(&meta("roy", "424242")), |_| None);
        let LockState::Stale(ref holder) = state else {
            panic!("expected stale, got {state:?}");
        };
        assert_eq!(holder.owner, "roy");
        let f = field(&state, &state, Some("sebastian"));
        assert_eq!(f, LockField::Stale("roy".into()));
        assert!(!f.is_protected());
    }

    /// The defect the agreement test against `hostlock.sh` caught in this very
    /// module: `/proc/<pid>` still exists for a zombie, so an existence check
    /// reports a held lock on a corpse forever. The script calls this "the
    /// common shape, not the exotic one" because every agent harness here
    /// launches long commands without an immediate `wait()`.
    #[test]
    fn a_zombie_anchor_is_dead_despite_having_a_proc_entry() {
        let state = classify(Some(&meta("roy", "7")), |_| {
            Some(ProcInfo {
                state: 'Z',
                start_time: 900,
                threads: Some(1),
            })
        });
        assert!(
            matches!(state, LockState::Stale(_)),
            "a zombie anchor must not hold the box forever, got {state:?}"
        );
    }

    /// ...but state `Z` is not proof of death. A thread-group leader that
    /// exited via `pthread_exit` while its threads keep running reports `Z` for
    /// a fully live process, and reaping that one takes a live holder's machine
    /// mid-benchmark -- the worse of the two errors by a wide margin.
    #[test]
    fn a_zombie_leader_with_live_threads_is_not_treated_as_dead() {
        let state = classify(Some(&meta("roy", "7")), |_| {
            Some(ProcInfo {
                state: 'Z',
                start_time: 900,
                threads: Some(6),
            })
        });
        assert!(matches!(state, LockState::Held(_)), "got {state:?}");
        // No readable `Threads:` is no evidence of death, so it must not be
        // read as one. There is deliberately no numeric default.
        let state = classify(Some(&meta("roy", "7")), |_| {
            Some(ProcInfo {
                state: 'Z',
                start_time: 900,
                threads: None,
            })
        });
        assert!(matches!(state, LockState::Held(_)), "got {state:?}");
    }

    /// The other case an existence check gets wrong: some unrelated process was
    /// handed the same PID number, and the lock reads as current half an hour
    /// after its owner exited.
    #[test]
    fn a_recycled_pid_is_not_mistaken_for_the_original_holder() {
        let state = classify(Some(&meta("roy", "7")), |_| running(1234));
        assert!(
            matches!(state, LockState::Stale(_)),
            "a different process with the same pid is not the holder, got {state:?}"
        );
        assert!(matches!(
            classify(Some(&meta("roy", "7")), |_| running(900)),
            LockState::Held(_)
        ));
    }

    /// An anchor that cannot be checked is unproven, not dead, and equally not
    /// alive. Collapsing it either way is a decision made on absent evidence.
    #[test]
    fn an_unverifiable_anchor_is_neither_held_nor_stale() {
        // Live pid, no recorded start time: the script's
        // `unverifiable_live_anchor`.
        let state = classify(Some("owner=roy\nanchor_pid=7\nreason=acc0\n"), |_| {
            running(900)
        });
        assert_eq!(
            state,
            LockState::Unverified(LockHolder {
                owner: "roy".into(),
                owner_raw: "roy".into(),
                anchor_pid: Some(7),
                start_time: None,
                reason: "acc0".into(),
            })
        );
        let f = field(&state, &state, Some("roy"));
        assert_eq!(f, LockField::Unverified("roy".into()));
        assert!(
            !f.is_protected(),
            "unproven liveness must not certify a row even when the owner matches"
        );
        // No anchor at all is the same class of ignorance.
        assert!(matches!(
            classify(Some("owner=roy\n"), |_| running(900)),
            LockState::Unverified(_)
        ));
    }

    /// Sanitising is lossy, so it must not be what decides whether a lock is
    /// ours. Both collisions below would otherwise promote another agent's
    /// declaration to `mine` and mark a contaminated row protected -- the one
    /// direction this module exists to prevent.
    #[test]
    fn a_name_that_merely_sanitises_to_ours_is_not_ours() {
        let punctuated = parse_meta(&meta("sebastian!", "1")).expect("present");
        assert_eq!(
            punctuated.owner, "sebastian",
            "the display form is expected to collide; that is what makes this a hazard"
        );
        let state = LockState::Held(punctuated);
        assert_eq!(
            field(&state, &state, Some("sebastian")),
            LockField::Foreign("sebastian".into()),
            "a different raw owner must stay foreign however it prints"
        );
        assert!(!field(&state, &state, Some("sebastian")).is_protected());

        // Two names that differ only past the display truncation.
        let long = format!("{}a", "s".repeat(MAX_OWNER_LEN));
        let other = format!("{}b", "s".repeat(MAX_OWNER_LEN));
        let state = LockState::Held(parse_meta(&meta(&long, "1")).expect("present"));
        assert!(
            !field(&state, &state, Some(&other)).is_protected(),
            "truncation must not make two distinct owners the same owner"
        );
        assert!(field(&state, &state, Some(&long)).is_protected());
    }

    /// Every agent on this host runs as the same user, so a `$USER` fallback
    /// would hand one agent another's declaration. Absent attribution has to
    /// stay absent.
    #[test]
    fn an_absent_owner_is_never_filled_in_from_somewhere_else() {
        let f = field(&held("roy"), &held("roy"), None);
        assert!(!f.is_protected());
        assert_eq!(f, LockField::Held("roy".into()));
        // An empty or whitespace-only HOSTLOCK_OWNER is not an attribution
        // either, and must not match a holder whose owner sanitises to "?".
        let blank = LockState::Held(parse_meta(&meta("   ", "1")).expect("present"));
        assert!(
            !field(&blank, &blank, Some("   ")).is_protected(),
            "two unnamed parties are not the same party"
        );
    }

    /// A non-numeric anchor must not fall back to a PID that happens to exist.
    #[test]
    fn a_corrupt_anchor_is_not_rounded_to_a_live_pid() {
        let holder = parse_meta("owner=roy\nanchor_pid=not-a-pid\n").expect("owner is present");
        assert_eq!(holder.anchor_pid, None);
        let holder = parse_meta("owner=roy\nanchor_pid=7\nstart_time=nonsense\n").expect("present");
        assert_eq!(holder.start_time, None);
    }

    /// A lock directory that cannot be read is not an empty one. Laundering a
    /// permission error into `free` would put a clean label on every row taken
    /// on a machine whose lock this process simply could not see.
    #[test]
    fn an_unreadable_lock_directory_is_not_reported_as_an_absent_one() {
        use std::io::ErrorKind;
        assert_eq!(
            classify_io(Err(ErrorKind::NotFound), |_| running(900)),
            LockState::Free
        );
        for kind in [
            ErrorKind::PermissionDenied,
            ErrorKind::IsADirectory,
            ErrorKind::InvalidData,
        ] {
            assert_eq!(
                classify_io(Err(kind), |_| running(900)),
                LockState::Unknown,
                "{kind:?} is not evidence that the lock is free"
            );
        }
    }

    /// Absent metadata is a measurement; unparseable metadata is not.
    #[test]
    fn a_half_written_lock_is_unknown_and_an_absent_one_is_free() {
        assert_eq!(classify(None, |_| running(900)), LockState::Free);
        assert_eq!(
            classify(Some("acquired_at=now\n"), |_| running(900)),
            LockState::Unknown
        );
    }

    /// Agreeing with `hostlock.sh`'s `head -1` matters more than picking the
    /// better rule, and only differs under the corrupt metadata where it counts.
    #[test]
    fn duplicate_keys_resolve_the_same_way_the_writer_resolves_them() {
        let holder = parse_meta("owner=roy\nowner=sebastian\nanchor_pid=1\n").expect("present");
        assert_eq!(holder.owner, "roy");
    }

    /// A key must match whole, or `downstream_owner=x` would answer for `owner`.
    #[test]
    fn a_key_is_matched_at_the_start_of_a_line_only() {
        assert_eq!(
            meta_get("downstream_owner=x\nowner=roy\n", "owner"),
            Some("roy")
        );
        assert_eq!(meta_get("ownership=x\n", "owner"), None);
        // `start_time` and `time` must not answer for one another either.
        assert_eq!(meta_get("start_time=5\n", "time"), None);
    }

    /// The config parser is the half of the path rule that can drift silently:
    /// a reader that resolves a different directory from the writer reports
    /// `free` on a declared host, convincingly, forever.
    #[test]
    fn the_config_parser_matches_the_scripts_sed_expression() {
        // Whitespace on both sides of `=`, and a trailing comment, are not
        // part of the path.
        assert_eq!(
            parse_conf_lock_dir("  lock_dir = /var/lib/hl   # box-wide\n").as_deref(),
            Some("/var/lib/hl")
        );
        // First key wins, as `head -1` does on the other side.
        assert_eq!(
            parse_conf_lock_dir("lock_dir=/a\nlock_dir=/b\n").as_deref(),
            Some("/a")
        );
        // A key must match whole: `lock_dir_old=` is a different key, and
        // answering for it would move the lock on the strength of a comment.
        assert_eq!(parse_conf_lock_dir("lock_dir_old=/a\n"), None);
        assert_eq!(parse_conf_lock_dir("# lock_dir=/a\n"), None);
        assert_eq!(parse_conf_lock_dir("other=1\n"), None);
        // ASCII whitespace only, matching `sed`'s `[[:space:]]`. A NO-BREAK
        // SPACE is part of the path on both sides, or the two implementations
        // resolve different directories from one file.
        assert_eq!(
            parse_conf_lock_dir("lock_dir=/var/lib/hl\u{a0}\n").as_deref(),
            Some("/var/lib/hl\u{a0}")
        );
        // Present but empty is Some(""), NOT None: absent means "no opinion,
        // use the default", empty means "the admin tried to say something and
        // it is unusable". Collapsing them would silently send this process to
        // /tmp while the rest of the box followed a config it could not parse.
        assert_eq!(parse_conf_lock_dir("lock_dir=\n").as_deref(), Some(""));
    }

    /// An unusable config must not be laundered into a location, nor into
    /// `free`: `read` answers `Unknown` for it, which is the arm that stops a
    /// row from being certified.
    #[test]
    fn an_unusable_config_is_an_error_rather_than_a_fallback() {
        assert!(
            parse_conf_lock_dir("lock_dir=relative/path\n").is_some_and(|v| !v.starts_with('/'))
        );
        let err = LockDirError {
            config: std::path::PathBuf::from("/etc/hostlock.conf"),
            value: "relative/path".to_string(),
        };
        assert!(err.to_string().contains("relative/path"));
        assert!(err.to_string().contains("absolute"));
    }

    #[test]
    fn a_reason_is_reported_only_when_both_readings_agree_on_a_holder() {
        let a = held("sebastian");
        assert_eq!(reason(&a, &a).as_deref(), Some("acc0"));
        assert_eq!(reason(&a, &held("roy")), None);
        assert_eq!(reason(&LockState::Free, &LockState::Free), None);
    }

    /// The states a run may be certified under, and the states it may not, must
    /// not print the same string. `free` and `unusable` are the pair worth
    /// pinning: both mean "no lock was found", and only one of them means the
    /// host was idle.
    #[test]
    fn an_unusable_host_does_not_print_as_a_free_one() {
        assert_eq!(LockField::Free.to_string(), "free");
        assert_eq!(LockField::Unusable.to_string(), "unusable");
        assert_ne!(LockField::Unusable, LockField::Free);
        assert!(!LockField::Unusable.is_protected());
        assert_eq!(
            field(&LockState::Unusable, &LockState::Unusable, Some("leon")),
            LockField::Unusable,
            "a name cannot certify a row on a host that has no lock to hold"
        );
        // A host that becomes usable mid-window changed custody in the only
        // sense that matters: the second half could have been declared and the
        // first half could not.
        assert_eq!(
            field(&LockState::Unusable, &LockState::Free, None),
            LockField::Changed
        );
    }
}
