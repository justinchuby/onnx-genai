//! Checks the reader against the writer, by running the real
//! `scripts/hostlock.sh`.
//!
//! The unit tests in `hostlock.rs` are all pure functions over strings I wrote
//! myself, so they prove the decision table is internally consistent and prove
//! nothing at all about whether it describes the file `hostlock.sh` actually
//! writes. A reader whose key names had drifted from the script would pass every
//! one of them and then report `free` on a locked host forever -- convincingly,
//! and in exactly the situation the field exists to catch.
//!
//! So this test does not parse a fixture. It shells out to the script, has it
//! take a lock, and asserts the Rust side sees what the shell side just did.
//!
//! # Why it fails rather than skips when the script is missing
//!
//! A skip would be the same defect one level up: the run would print `ok`, the
//! agreement would be unchecked, and the output would be indistinguishable from
//! a real pass. The script is committed at a fixed path in this repository, so
//! its absence means the reader's assumptions about that path are stale --
//! which is precisely the thing worth failing on. The platform gate is
//! `#[cfg]`, so on a non-Linux target these tests do not exist rather than
//! passing vacuously.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn script() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/hostlock.sh")
        .canonicalize()
        .expect(
            "scripts/hostlock.sh must exist: this test's whole purpose is to agree with it, and \
             if it has moved then the reader's path assumptions have gone stale unnoticed",
        );
    assert!(path.is_file(), "{} is not a file", path.display());
    path
}

/// A lock directory under `CARGO_TARGET_TMPDIR`, so the test never touches the
/// real one. Sharing the default path would let a test release a lock a
/// colleague was relying on -- a test that can hand somebody else's machine
/// away is worse than no test.
fn lock_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn hostlock(dir: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new("bash")
        .arg(script())
        .args(args)
        .env("HOSTLOCK_DIR", dir)
        // Acknowledge the private lock. Every invocation here sets
        // `HOSTLOCK_DIR`, which the script now announces on stderr as
        // coordinating with nobody -- correct, deliberate, and exactly what a
        // test wants, but this helper folds stderr into the text it asserts on.
        .env("HOSTLOCK_PRIVATE_OK", "1")
        .output()
        .expect("failed to run scripts/hostlock.sh");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// Kills and reaps its child on unwind.
///
/// Without this, a panic between spawning the anchor and killing it leaves a
/// `sleep 300` reparented to init for five minutes. It burns no CPU, so it
/// would not corrupt anyone's measurement -- but complaining about other
/// agents' leaked processes while leaking one on every failed assertion is not
/// a position worth defending, and the failing run is exactly when someone will
/// be looking at the process table.
struct Anchor(std::process::Child);

impl Anchor {
    fn spawn() -> Self {
        Anchor(
            Command::new("sleep")
                .arg("300")
                .spawn()
                .expect("spawn anchor"),
        )
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }

    /// Ends the anchor and waits for it, so the PID names nothing afterwards
    /// rather than a zombie.
    fn retire(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for Anchor {
    fn drop(&mut self) {
        self.retire();
    }
}

/// Removes a scratch lock directory on unwind. Never the real one: it is
/// constructed only by [`lock_dir`], which roots everything under
/// `CARGO_TARGET_TMPDIR`.
struct ScratchLock(PathBuf);

impl Drop for ScratchLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_at(dir: &Path) -> onnx_runtime_hostmon::hostlock::LockState {
    // Deliberately not `hostlock::read()`: that reads `HOSTLOCK_DIR` from the
    // process environment, and a test that mutated it would leak the override
    // into every other test in this binary on any failing assertion. Same file
    // contents, same classifier, no shared mutable state.
    let meta = std::fs::read_to_string(dir.join("meta")).ok();
    onnx_runtime_hostmon::hostlock::classify(
        meta.as_deref(),
        onnx_runtime_hostmon::hostlock::proc_info,
    )
}

/// The load-bearing case: the script writes, the reader reads, and they agree
/// on who holds the lock, on when it stops being held, and on the fact that a
/// live holder's lock cannot be taken away.
#[test]
fn the_reader_sees_the_lock_the_script_just_took() {
    use onnx_runtime_hostmon::hostlock::{LockField, LockState, field};

    let dir = ScratchLock(lock_dir("hostlock-agreement"));
    let dir = &dir.0;

    assert_eq!(
        read_at(dir),
        LockState::Free,
        "an absent lock directory must read as free, not as unknown"
    );

    // Anchored to a child this test owns and can kill. Anchoring to the
    // script's own shell would make the lock stale the instant it returned, so
    // the test would assert the staleness path while claiming to assert the
    // held one.
    let mut anchor = Anchor::spawn();
    let anchor_pid = anchor.pid();

    let (ok, out) = hostlock(
        dir,
        &[
            "acquire",
            "--owner",
            "sebastian",
            "--reason",
            "agreement-test",
            "--pid",
            &anchor_pid.to_string(),
        ],
    );
    assert!(ok, "acquire failed: {out}");

    let held = read_at(dir);
    let LockState::Held(ref holder) = held else {
        panic!("the reader must see the lock the script just wrote, got {held:?}");
    };
    assert_eq!(
        holder.owner, "sebastian",
        "owner key disagrees with the writer"
    );
    assert_eq!(
        holder.anchor_pid,
        Some(anchor_pid),
        "anchor_pid key disagrees with the writer"
    );
    assert_eq!(
        holder.reason, "agreement-test",
        "reason key disagrees with the writer"
    );
    assert!(
        holder.start_time.is_some(),
        "start_time must be read, or a recycled pid reads as the original holder"
    );

    assert_eq!(
        field(&held, &held, Some("sebastian")),
        LockField::Mine("sebastian".into())
    );
    assert_eq!(
        field(&held, &held, Some("roy")),
        LockField::Foreign("sebastian".into()),
        "a lock held by someone else must never mark a row protected"
    );

    // The script refuses to release a lock whose anchor is still running. That
    // refusal is the property that makes the lock worth reading at all, so it is
    // asserted rather than stepped around with HOSTLOCK_FORCE.
    let (ok, out) = hostlock(dir, &["release", "--owner", "sebastian"]);
    assert!(
        !ok,
        "a lock with a live anchor must not be releasable by a bystander: {out}"
    );
    assert!(
        matches!(read_at(dir), LockState::Held(_)),
        "a refused release must leave the lock exactly as it was"
    );

    anchor.retire();

    assert!(
        matches!(read_at(dir), LockState::Stale(_)),
        "once the anchor is gone and reaped, the lock is stale"
    );

    let (ok, out) = hostlock(dir, &["release", "--owner", "sebastian"]);
    assert!(ok, "release of a dead-anchor lock failed: {out}");
    assert_eq!(
        read_at(dir),
        LockState::Free,
        "release must return the lock to free"
    );

    // A run spanning the release is the case a single end-of-window reading
    // would have reported as a clean `mine:sebastian`.
    assert_eq!(
        field(&held, &read_at(dir), Some("sebastian")),
        LockField::Changed
    );
}

/// A zombie anchor: `/proc/<pid>` still exists and its start time still
/// matches, so the obvious liveness check reports a held lock on a corpse
/// forever. This reader had exactly that defect until this test found it.
///
/// `hostlock.sh` calls this the common shape rather than the exotic one,
/// because every agent harness on this box launches long commands without an
/// immediate `wait()`. It is reproduced here for real -- a spawned child that
/// is never reaped -- rather than simulated with a fabricated `ProcInfo`, so
/// that it also confirms `proc_info` parses a genuine `/proc/<pid>/stat`.
#[test]
fn the_reader_and_the_script_agree_that_a_zombie_anchor_is_dead() {
    use onnx_runtime_hostmon::hostlock::{LockField, LockState, field};

    let dir = ScratchLock(lock_dir("hostlock-zombie"));
    let dir = &dir.0;

    let mut corpse = Command::new("true").spawn().expect("spawn");
    let pid = corpse.id();
    // Wait for it to exit without reaping it, which is what makes it a zombie
    // rather than a departed process. `try_wait` would reap it.
    for _ in 0..500 {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        if stat
            .rsplit(')')
            .next()
            .is_some_and(|r| r.trim_start().starts_with('Z'))
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let stat =
        std::fs::read_to_string(format!("/proc/{pid}/stat")).expect("zombie has a /proc entry");
    assert!(
        stat.rsplit(')')
            .next()
            .is_some_and(|r| r.trim_start().starts_with('Z')),
        "the fixture must actually be a zombie, or this test proves nothing: {stat}"
    );

    let (ok, out) = hostlock(
        dir,
        &["acquire", "--owner", "roy", "--pid", &pid.to_string()],
    );
    assert!(ok, "acquire failed: {out}");

    let state = read_at(dir);
    let LockState::Stale(ref holder) = state else {
        panic!("a zombie anchor must not hold the box forever, got {state:?}");
    };
    assert_eq!(holder.owner, "roy");

    let f = field(&state, &state, Some("sebastian"));
    assert_eq!(f, LockField::Stale("roy".into()));
    assert!(
        !f.is_protected(),
        "a stale lock does not stop the orphaned load it was covering, so it must not certify a row"
    );

    // The script must reach the same verdict, otherwise a reaper and a
    // published row would tell an operator two different stories about one lock.
    let (_, status) = hostlock(dir, &["status", "--porcelain"]);
    assert!(
        status.to_lowercase().contains("stale"),
        "the script must call this lock stale too, said: {status}"
    );

    corpse.wait().expect("reap");
}

/// The two implementations must look in the same place when nothing redirects
/// them.
///
/// Every other test in this file overrides `HOSTLOCK_DIR`, because a test that
/// could release a colleague's lock is worse than no test -- which means the
/// default path, the one every real run uses, is the single value that all the
/// other agreement checks are structurally unable to see. If the script's
/// default moved and the reader's did not, the reader would find no metadata
/// file, report `free`, and every row would carry a confident `host_lock=free`
/// on a locked host. That failure is silent in the direction that permits a
/// run, so it is asserted here rather than trusted.
///
/// Asked of the script's BEHAVIOUR, not of its text. Since the path became a
/// resolution (env, then config, then default) rather than a single
/// assignment, no `LOCK_DIR=` line is the answer on its own, and a text scan
/// would have to model shell control flow to know which one wins. `status`
/// reads and never creates, so asking it about the real default cannot take,
/// modify or leave a lock on the actual host. The constant is pinned as well,
/// so a rename is still caught rather than silently checking nothing.
#[test]
fn the_default_lock_dir_matches_the_script() {
    let (ok, out) = hostlock_conf(
        Path::new("/nonexistent/onnx-genai/hostlock.conf"),
        &["status", "--porcelain"],
    );
    assert!(ok, "status failed: {out}");
    let seen = out
        .lines()
        .find_map(|l| l.strip_prefix("lock_dir="))
        .unwrap_or_else(|| panic!("status --porcelain must say which lock it measured: {out}"));
    assert_eq!(
        seen,
        onnx_runtime_hostmon::hostlock::DEFAULT_LOCK_DIR,
        "the reader looks in a different directory than the script writes to, so it would report \
         `free` on a held host"
    );
    assert!(
        out.contains("lock_dir_source=default"),
        "with no env override and no config, the path must be attributed to the default, or this \
         test is comparing against whatever this host happens to be configured for: {out}"
    );

    let text = std::fs::read_to_string(script()).expect("read hostlock.sh");
    let assignments: Vec<&str> = text
        .lines()
        .filter(|l| l.trim_start().starts_with("HOSTLOCK_BUILTIN_DIR="))
        .collect();
    // Exactly one, not "the first one". Shell takes the last assignment that
    // executes; a scan takes the first that appears. Where those differ the
    // test would read a decoy and pass while the two implementations disagreed
    // -- checking nothing, and reporting it as agreement. Rather than model
    // shell control flow, refuse to answer when the question is ambiguous.
    let n = assignments.len();
    assert_eq!(
        n, 1,
        "hostlock.sh must assign HOSTLOCK_BUILTIN_DIR exactly once, found {n}: {assignments:?}. \
         More than one assignment means this test cannot tell which default is the effective one, \
         and guessing would let it pass while the reader looked somewhere the script never writes",
    );
    let default = assignments[0]
        .trim()
        .split_once('=')
        .map(|(_, value)| value.trim().to_string())
        .expect("HOSTLOCK_BUILTIN_DIR must have a value");
    assert_eq!(
        default,
        onnx_runtime_hostmon::hostlock::DEFAULT_LOCK_DIR,
        "the script's built-in default and the reader's constant have diverged"
    );
}

/// Runs the script with the machine-local config, and *without* `HOSTLOCK_DIR`,
/// so the path under test is the one it resolves for itself.
///
/// `status` is the only subcommand used: it reads and never creates, so a
/// resolution that came out wrong cannot leave a lock behind on a real path.
fn hostlock_conf(conf: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new("bash")
        .arg(script())
        .args(args)
        .env_remove("HOSTLOCK_DIR")
        .env("HOSTLOCK_CONF", conf)
        .output()
        .expect("failed to run scripts/hostlock.sh");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

fn write_conf(name: &str, text: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, text).expect("write config");
    path
}

/// The two implementations must agree on *which directory* the lock is in.
///
/// This is the drift that is invisible until it matters: a reader resolving
/// `/tmp` on a host whose config moved the lock finds no metadata, reports
/// `free`, and stamps that on every row of a run taken while a peer held the
/// box. No assertion in either codebase fails, and the reassuring answer is the
/// wrong one. So the parsing rules are compared against the script itself
/// rather than against my reading of the `sed` expression.
#[test]
fn the_reader_and_the_script_resolve_the_same_configured_directory() {
    use onnx_runtime_hostmon::hostlock::{parse_conf_lock_dir, resolve_lock_dir_from};

    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let target = base.join("configured-lock");
    let target = target.to_str().expect("utf-8 path");

    let cases: Vec<(String, String)> = vec![
        (format!("lock_dir={target}\n"), target.to_string()),
        // Whitespace around `=` and a trailing comment are not part of the path.
        (
            format!("  lock_dir =  {target}   # box-wide\n"),
            target.to_string(),
        ),
        // First key wins on both sides.
        (
            format!("lock_dir={target}\nlock_dir=/somewhere/else\n"),
            target.to_string(),
        ),
        // A commented-out key is not a key, and must not move the lock.
        (
            format!("# lock_dir={target}\nlock_dir={target}\n"),
            target.to_string(),
        ),
    ];

    for (i, (text, want)) in cases.iter().enumerate() {
        let conf = write_conf(&format!("hostlock-conf-{i}.conf"), text);
        let (ok, out) = hostlock_conf(&conf, &["status", "--porcelain"]);
        assert!(ok, "status failed for config {text:?}: {out}");
        let seen = out
            .lines()
            .find_map(|l| l.strip_prefix("lock_dir="))
            .unwrap_or_else(|| {
                panic!(
                    "status --porcelain must say which lock it measured, or a row \
                     cannot be re-checked; got: {out}"
                )
            });
        assert_eq!(seen, want, "the script resolved {seen} for {text:?}");
        assert_eq!(
            parse_conf_lock_dir(text).as_deref(),
            Some(want.as_str()),
            "the reader resolved a different directory from the writer for {text:?}"
        );
        assert!(
            out.contains("lock_dir_source=config"),
            "a configured path must be attributed to the config, not the default: {out}"
        );
        // The rule `read()` actually uses, not just the parser it calls. A
        // reader whose resolution ignored the config would satisfy every
        // assertion above and still look at the wrong directory.
        assert_eq!(
            resolve_lock_dir_from(None, &conf).expect("usable config"),
            PathBuf::from(want),
            "the reader resolved a different directory from the writer for {text:?}"
        );
        // And the env override outranks it on both sides, because it is set per
        // process and therefore cannot be a box-wide decision.
        assert_eq!(
            resolve_lock_dir_from(Some("/elsewhere"), &conf).expect("env wins"),
            PathBuf::from("/elsewhere")
        );
    }

    // A key that merely starts the same way is a different key on both sides.
    let conf = write_conf(
        "hostlock-conf-otherkey.conf",
        &format!("lock_dir_old={target}\n"),
    );
    let (ok, out) = hostlock_conf(&conf, &["status", "--porcelain"]);
    assert!(ok, "status failed: {out}");
    assert!(
        out.contains("lock_dir=/tmp/onnx-genai-hostlock"),
        "an unrelated key must leave the default path alone: {out}"
    );
    assert_eq!(
        parse_conf_lock_dir(&format!("lock_dir_old={target}\n")),
        None
    );
    assert_eq!(
        resolve_lock_dir_from(None, &conf).expect("no usable key"),
        PathBuf::from("/tmp/onnx-genai-hostlock"),
        "an unrelated key must leave the reader on the default path too"
    );
    // A config that is not there at all is not an error: it is the default.
    assert_eq!(
        resolve_lock_dir_from(None, Path::new("/nonexistent/hostlock.conf")).expect("absent"),
        PathBuf::from("/tmp/onnx-genai-hostlock")
    );
}

/// A config that cannot be honoured must stop the run, not silently send this
/// process to the default while the rest of the box follows the config.
///
/// Half a host on one path and half on the other is worse than either choice:
/// both halves acquire instantly, neither ever collides, and every row claims a
/// declared host.
#[test]
fn neither_implementation_falls_back_when_the_config_is_unusable() {
    use onnx_runtime_hostmon::hostlock::parse_conf_lock_dir;

    for (i, text) in ["lock_dir=relative/path\n", "lock_dir=\n"]
        .iter()
        .enumerate()
    {
        let conf = write_conf(&format!("hostlock-conf-bad-{i}.conf"), text);
        let (ok, out) = hostlock_conf(&conf, &["status", "--porcelain"]);
        assert!(!ok, "an unusable config must fail the command: {out}");
        assert!(
            !out.contains("lock_dir=/tmp/onnx-genai-hostlock"),
            "it must not fall back to the default path: {out}"
        );
        // The reader's half of the same rule: present but not absolute is an
        // error, and `read()` turns it into `Unknown` rather than `Free`.
        let value = parse_conf_lock_dir(text).expect("the key is present");
        assert!(
            !value.starts_with('/'),
            "the reader must not accept {value:?} as a lock directory"
        );
        let err = onnx_runtime_hostmon::hostlock::resolve_lock_dir_from(None, &conf)
            .expect_err("an unusable config must not resolve to a directory");
        assert_eq!(err.value, value);
    }
}

/// A window that the script took a lock inside of is reported as `changed`,
/// not as a holder.
///
/// The unit tests for this hand `Window` two states I wrote by hand, so they
/// prove the rule and prove nothing about whether either state can arise from
/// the real script. Here both readings come from `hostlock.sh` -- the first
/// from a directory it has not touched, the second from one it has just
/// acquired -- which is the shape of the error the two-ended read exists to
/// catch: a benchmark that starts on a free host and finishes on a claimed one.
#[test]
fn a_window_the_script_claimed_midway_is_reported_as_changed() {
    let dir = lock_dir("window-changed");
    let _cleanup = ScratchLock(dir.clone());
    let mut anchor = Anchor::spawn();

    let before = read_at(&dir);
    assert_eq!(
        before,
        onnx_runtime_hostmon::hostlock::LockState::Free,
        "the scratch directory must start unlocked, or this test proves nothing"
    );

    let (ok, out) = hostlock(
        &dir,
        &[
            "acquire",
            "--owner",
            "someone-else",
            "--reason",
            "mid-window claim",
            "--pid",
            &anchor.pid().to_string(),
        ],
    );
    assert!(ok, "acquire failed: {out}");

    let after = read_at(&dir);
    let report =
        onnx_runtime_hostmon::window::Window::opened_at(before).close_as(None, || after.clone());
    assert_eq!(
        report.field(),
        &onnx_runtime_hostmon::hostlock::LockField::Changed
    );
    assert_eq!(
        report.to_string(),
        "host_lock=changed",
        "a changed window must not carry the late holder's reason: naming one invites a reader to \
         treat the window as covered after all"
    );
    assert!(!report.is_protected());
    assert!(
        report
            .warning()
            .is_some_and(|w| w.starts_with("UNPROTECTED")),
        "a window that changed hands must complain"
    );

    // The steady case, from the same real lock: both ends agree, so the
    // holder's own reason survives into the row.
    let steady = onnx_runtime_hostmon::window::Window::opened_at(after.clone())
        .close_as(None, || after.clone());
    // The reason was written with a space in it and comes back with an
    // underscore: the row is a `key=value` list, so a reason that kept its
    // spaces would add fields to it. Asserted against a reason the *script*
    // wrote, not one built in-process, because that is the path an injected
    // reason would actually travel.
    assert_eq!(
        steady.to_string(),
        "host_lock=held:someone-else lock_reason=mid-window_claim",
        "with HOSTLOCK_OWNER unset there is nothing to attribute the lock against, so `held:` is \
         the honest answer and it is unprotected either way"
    );
    assert_eq!(
        steady.to_string().split_whitespace().count(),
        2,
        "a holder's reason must not be able to add fields to a result row"
    );
    assert!(!steady.is_protected());

    anchor.retire();
}

/// Set on a child of this test binary to tell it which probe to run.
const PROBE_MODE: &str = "HOSTMON_WINDOW_PROBE";

/// Runs one probe in a child process and returns everything it printed.
///
/// A child rather than an in-process call because [`Window::open`] and
/// [`Window::close`] read `HOSTLOCK_DIR` and `HOSTLOCK_OWNER` from the process
/// environment, and there is no way to point them at a scratch lock from inside
/// this binary without a process-wide override that leaks into every later test
/// on any failing assertion -- the defect #1591 spent a PR removing. A child
/// gets its own environment, so the override dies with it.
///
/// `HOSTLOCK_OWNER` is *removed* rather than merely not set: this binary is run
/// by agents who export it, and inheriting it would make the assertions below
/// depend on whose shell started the run.
fn run_probe(mode: &str, dir: &Path, owner: Option<&str>) -> String {
    let exe = std::env::current_exe().expect("a test binary knows its own path");
    let mut cmd = Command::new(exe);
    cmd.args([
        "window_probe_child",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ])
    .env(PROBE_MODE, mode)
    .env("HOSTLOCK_DIR", dir);
    match owner {
        Some(owner) => cmd.env("HOSTLOCK_OWNER", owner),
        None => cmd.env_remove("HOSTLOCK_OWNER"),
    };
    let out = cmd.output().expect("spawn the probe child");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "probe child failed:\n{text}");
    // The filter has to have matched something. A renamed test would leave the
    // child passing with zero tests run, and the parent would then assert
    // against output that no probe produced -- a green run that measured
    // nothing, which is the failure this whole file exists to make impossible.
    assert!(
        text.contains("1 passed"),
        "the probe child ran no test, so its silence proves nothing:\n{text}"
    );
    text
}

/// Reads the one `PROBE ` line out of a child's output.
fn probe_row(text: &str) -> &str {
    // Not `strip_prefix`: libtest writes `test <name> ... ` before the first
    // line the test itself prints, so the marker lands mid-line.
    let rows: Vec<&str> = text
        .lines()
        .filter_map(|line| line.split_once("PROBE ").map(|(_, row)| row.trim_end()))
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "expected exactly one probe row, got {}:\n{text}",
        rows.len()
    );
    rows[0]
}

/// The body of the child. Does nothing unless [`PROBE_MODE`] says which arm to
/// run, which is only ever set by [`run_probe`].
///
/// It is not itself an assertion -- the parent tests make those. It is here
/// rather than in a separate binary because an integration test binary is the
/// only thing in this crate that is already built and already linked against
/// the library.
#[test]
fn window_probe_child() {
    let Ok(mode) = std::env::var(PROBE_MODE) else {
        return;
    };
    let dir = PathBuf::from(
        std::env::var("HOSTLOCK_DIR").expect("the parent points the child at a scratch lock"),
    );
    let report = match mode.as_str() {
        // The lock is already held, by the owner the parent exported. Both
        // readings come from `hostlock::read` via the environment.
        "held" => onnx_runtime_hostmon::window::Window::open().close(),
        // Free at the open, claimed by somebody else before the close.
        "changed" => {
            let window = onnx_runtime_hostmon::window::Window::open();
            let mut anchor = Anchor::spawn();
            let (ok, out) = hostlock(
                &dir,
                &[
                    "acquire",
                    "--owner",
                    "a-co-tenant",
                    "--reason",
                    "arrived midway",
                    "--pid",
                    &anchor.pid().to_string(),
                ],
            );
            assert!(ok, "acquire failed in the child: {out}");
            let report = window.close();
            anchor.retire();
            report
        }
        other => panic!("unknown probe mode {other:?}"),
    };
    println!(
        "PROBE {report} protected={} warned={}",
        report.is_protected(),
        report.warning().is_some()
    );
}

/// `Window::open` and `Window::close` against the real script, the real
/// `HOSTLOCK_DIR` and the real `HOSTLOCK_OWNER`.
///
/// Every other test of `Window` hands it states built in-process and an owner
/// passed as an argument, which is deliberate -- but it leaves the two
/// functions every benchmark actually calls covered by nothing. `open` reading
/// the wrong directory, or `close` reusing the first reading, or the owner
/// never reaching `field`, would pass the whole unit suite and put
/// `host_lock=free` on every row of a locked host.
///
/// So this drives them end to end, in a child, and asserts the exact row text a
/// benchmark would print.
#[test]
fn open_and_close_read_the_real_lock_and_the_real_owner() {
    let dir = lock_dir("window-probe");
    let _cleanup = ScratchLock(dir.clone());
    let mut anchor = Anchor::spawn();

    // A window this process held end to end: the one case that is protected and
    // the one case that must not warn.
    let (ok, out) = hostlock(
        &dir,
        &[
            "acquire",
            "--owner",
            "probe-owner",
            "--reason",
            "window probe",
            "--pid",
            &anchor.pid().to_string(),
        ],
    );
    assert!(ok, "acquire failed: {out}");

    let text = run_probe("held", &dir, Some("probe-owner"));
    assert_eq!(
        probe_row(&text),
        "host_lock=mine:probe-owner lock_reason=window_probe protected=true warned=false",
        "a lock this process declared, held across the whole window, is the only protected row"
    );

    // Same lock, same window, a different declared owner: the row must name it
    // as somebody else's rather than certify the run.
    let text = run_probe("held", &dir, Some("someone-else"));
    assert_eq!(
        probe_row(&text),
        "host_lock=foreign:probe-owner lock_reason=window_probe protected=false warned=true",
        "a co-tenant's lock must never certify a row"
    );

    // And with nothing declared, the honest answer is an unattributed holder.
    let text = run_probe("held", &dir, None);
    assert_eq!(
        probe_row(&text),
        "host_lock=held:probe-owner lock_reason=window_probe protected=false warned=true"
    );

    // The window that changed hands, from the real script, through the real
    // `close`. The reason the late holder wrote must not appear.
    //
    // A second scratch directory rather than releasing the first: the script
    // refuses to release a lock whose anchor is still alive, and the override
    // for that is `HOSTLOCK_FORCE=1` -- a test that reaches for a safety
    // bypass to arrange its own fixture is training the habit that loses
    // somebody a run.
    let changed_dir = lock_dir("window-probe-changed");
    let _changed_cleanup = ScratchLock(changed_dir.clone());
    let text = run_probe("changed", &changed_dir, Some("probe-owner"));
    assert_eq!(
        probe_row(&text),
        "host_lock=changed protected=false warned=true",
        "a window that changed hands names no holder, whichever end held one"
    );

    anchor.retire();
}
