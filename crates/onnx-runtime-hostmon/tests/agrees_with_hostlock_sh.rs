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
