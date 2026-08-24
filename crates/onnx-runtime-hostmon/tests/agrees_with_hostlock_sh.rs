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
        .output()
        .expect("failed to run scripts/hostlock.sh");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
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

    let dir = lock_dir("hostlock-agreement");

    assert_eq!(
        read_at(&dir),
        LockState::Free,
        "an absent lock directory must read as free, not as unknown"
    );

    // Anchored to a child this test owns and can kill. Anchoring to the
    // script's own shell would make the lock stale the instant it returned, so
    // the test would assert the staleness path while claiming to assert the
    // held one.
    let mut anchor = Command::new("sleep")
        .arg("300")
        .spawn()
        .expect("spawn anchor");
    let anchor_pid = anchor.id();

    let (ok, out) = hostlock(
        &dir,
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

    let held = read_at(&dir);
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
    let (ok, out) = hostlock(&dir, &["release", "--owner", "sebastian"]);
    assert!(
        !ok,
        "a lock with a live anchor must not be releasable by a bystander: {out}"
    );
    assert!(
        matches!(read_at(&dir), LockState::Held(_)),
        "a refused release must leave the lock exactly as it was"
    );

    anchor.kill().expect("kill anchor");
    anchor.wait().expect("reap anchor");

    assert!(
        matches!(read_at(&dir), LockState::Stale(_)),
        "once the anchor is gone and reaped, the lock is stale"
    );

    let (ok, out) = hostlock(&dir, &["release", "--owner", "sebastian"]);
    assert!(ok, "release of a dead-anchor lock failed: {out}");
    assert_eq!(
        read_at(&dir),
        LockState::Free,
        "release must return the lock to free"
    );

    // A run spanning the release is the case a single end-of-window reading
    // would have reported as a clean `mine:sebastian`.
    assert_eq!(
        field(&held, &read_at(&dir), Some("sebastian")),
        LockField::Changed
    );

    let _ = std::fs::remove_dir_all(&dir);
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

    let dir = lock_dir("hostlock-zombie");

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
        &dir,
        &["acquire", "--owner", "roy", "--pid", &pid.to_string()],
    );
    assert!(ok, "acquire failed: {out}");

    let state = read_at(&dir);
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
    let (_, status) = hostlock(&dir, &["status", "--porcelain"]);
    assert!(
        status.to_lowercase().contains("stale"),
        "the script must call this lock stale too, said: {status}"
    );

    corpse.wait().expect("reap");
    let _ = std::fs::remove_dir_all(&dir);
}
