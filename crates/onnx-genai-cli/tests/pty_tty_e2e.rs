//! PTY-driven end-to-end tests for the terminal-only rendering behaviour.
//!
//! # Why PTY tests exist
//!
//! Two rendering decisions in `output.rs` and `live_turn.rs` branch on
//! `is_terminal()`:
//!
//! 1. **Trailing newline** — after a streamed reply whose last token does not
//!    end with `\n`, exactly one newline is written to separate the reply from
//!    the next prompt.  The write is skipped when the reply already ends with
//!    `\n` so a user never sees a blank line where there should be none.
//!
//! 2. **Stats format** — per-turn stats go to stderr; the format is a compact
//!    single line when stderr is piped and a two-line block when stderr is a
//!    terminal.  Selecting the format by probing *stdout* instead (the bug
//!    fixed in #372) inverts the choice whenever stdout and stderr are attached
//!    to different things.
//!
//! Piped-stdio tests structurally cannot exercise these branches because
//! `is_terminal()` returns `false` on every pipe.  Driving the binary under a
//! pseudo-terminal makes `is_terminal()` return `true` for whichever streams
//! are connected to a PTY slave, so both branches execute for real.
//!
//! # Platform gate
//!
//! These tests are `#[cfg(unix)]`.  ConPTY on Windows is disproportionately
//! complex to wire up in a test harness, and CI already runs Linux and macOS
//! jobs, so a Unix gate provides adequate coverage.  Note the precedent from
//! #298: tests are un-gated when the reason is incidental, gated when the
//! reason is genuine — PTY availability is a genuine one.

#[cfg(unix)]
mod pty_tty {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use std::io::Read;
    use std::os::fd::OwnedFd;
    use std::os::unix::io::AsFd;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    use nix::pty::{OpenptyResult, Winsize, openpty};
    use nix::unistd::dup;

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn text_model() -> PathBuf {
        repository_root().join("tests/fixtures/tiny-llm")
    }

    /// Open a fresh PTY pair.  The slave is the child-side terminal device;
    /// the master is the parent-side pipe into/out of that device.
    ///
    /// The slave is given a real 24×80 window size on purpose — **do not pass
    /// `None` here.**  `openpty(None, ..)` leaves the window at 0×0, and the
    /// `run` REPL's inline viewport (`live_turn.rs`) renders through ratatui's
    /// `insert_before_no_scrolling_regions`, which infinite-loops when the
    /// screen height is 0 (`to_draw = min(h, 0) = 0`, so it never makes
    /// forward progress).  A future type-ahead / streaming test that drives
    /// `run` (rather than the one-shot `generate` the tests below use) would
    /// hang the whole harness for the 30-second `drain_pty_master` timeout — and
    /// then time out CI — with a 0×0 window.  24×80 is the smallest ordinary
    /// terminal size that avoids that trap; the exact numbers are not magic,
    /// only "non-zero and realistic".
    ///
    /// Companion gotcha for anyone writing input into the master: a terminal
    /// sends **CR (`\r`)** for Enter, and crossterm maps CR → `Enter`.  Writing
    /// `\n` (LF) instead leaves the line un-terminated, so `read_line` never
    /// returns and the child hangs — another false "swallowed input".  Send
    /// `\r`, not `\n`.
    fn open_pty() -> (OwnedFd, OwnedFd) {
        let winsize = Winsize {
            ws_row: 24,
            ws_col: 80,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let OpenptyResult { master, slave } =
            openpty(Some(&winsize), None).expect("openpty must succeed on this platform");
        (master, slave)
    }

    /// Drain all bytes from a PTY master until EIO (the slave side is fully
    /// closed) or EOF, with a 30-second timeout.
    ///
    /// The timeout prevents a child that never closes the slave from blocking
    /// the test runner indefinitely.  When the timeout fires the function
    /// returns whatever bytes were collected; the test will then fail on its
    /// own assertions rather than hanging the CI job for hours.
    fn drain_pty_master(master: OwnedFd) -> Vec<u8> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut file: std::fs::File = master.into();
        let mut buf = [0u8; 4096];
        let mut collected = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            // poll() with a deadline instead of blocking read() so a child
            // that never exits cannot block the test harness forever.
            let ready = {
                let borrowed = file.as_fd();
                let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
                poll(
                    &mut fds,
                    PollTimeout::try_from(remaining).unwrap_or(PollTimeout::MAX),
                )
                .unwrap_or(0)
            };
            if ready == 0 {
                break; // timeout
            }
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => collected.extend_from_slice(&buf[..n]),
                // EIO when the last holder of the slave closes it (child exits).
                Err(_) => break,
            }
        }
        collected
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 — stats format is determined by *stderr*'s TTY state, not stdout's
    //
    // Setup
    //   stdin  = PTY slave  → is_terminal() = true  }  stats are enabled
    //   stdout = PTY slave  → is_terminal() = true  }  (both must be TTYs)
    //   stderr = pipe       → is_terminal() = false → single-line format
    //
    // The fix: emit_stats_line probes stderr.is_terminal(), not stdout.
    //
    // Pre-fix behaviour: stdout was probed (true) → block format → stderr
    // contained "]\n[" → this test would have FAILED.
    //
    // Post-fix behaviour: stderr is probed (false) → single-line → stderr
    // does NOT contain "]\n[" → this test PASSES.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn stats_single_line_when_stderr_is_piped_and_stdout_is_tty() {
        // One PTY covers both stdin and stdout (dup the slave so each stream
        // gets its own file-descriptor ownership).
        let (master, slave) = open_pty();
        let slave_stdin = dup(&slave).expect("dup must succeed");
        let slave_stdout = slave;

        let child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
            .arg("generate")
            .arg(text_model())
            .arg("--raw")
            .arg("--prompt")
            .arg("hi")
            .arg("--max-new-tokens")
            .arg("5")
            .env("ONNX_GENAI_EP", "cpu")
            // stdin + stdout → PTY slave (both is_terminal() = true → stats on)
            .stdin(Stdio::from(slave_stdin))
            .stdout(Stdio::from(slave_stdout))
            // stderr → pipe (is_terminal() = false → single-line stats expected)
            .stderr(Stdio::piped())
            .spawn()
            .expect("CLI binary must start");

        // Drain master (stdout of child) in a background thread so the child
        // is never blocked writing to the PTY.
        let drain = std::thread::spawn(move || drain_pty_master(master));

        let output = child.wait_with_output().expect("child must exit cleanly");
        let _ = drain.join();

        let stderr = String::from_utf8_lossy(&output.stderr);

        // Block stats contain the inter-line separator "]\n[" (two `[ … ]`
        // boxes joined by a newline).  Single-line stats do not.
        //
        // If this assertion fails the code is probing stdout (TTY) instead of
        // stderr (pipe) when selecting the stats format — the exact bug fixed
        // in #372.
        assert!(
            !stderr.contains("]\n["),
            "stats with stderr=pipe must use single-line format; \
             ']\\'\\n'[' indicates the block format was chosen by probing \
             stdout instead of stderr. stderr was: {stderr:?}"
        );

        // Sanity: stats must be present (not suppressed entirely).
        assert!(
            stderr.contains('[') && stderr.contains(']'),
            "stats must appear on stderr when stdin+stdout are TTYs; \
             got: {stderr:?}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 — exactly one trailing newline on TTY stdout after a streamed reply
    //
    // When stdout is a TTY the trailing newline is *conditional*:
    //   • added  when the reply's last token did not end with `\n`
    //   • omitted when it already did (otherwise a visible blank line appears)
    //
    // In both cases the output must end with exactly one CRLF (the PTY's
    // ONLCR flag maps every `\n` → `\r\n`).  A double CRLF (`\r\n\r\n`)
    // would be the blank-line defect this code path was introduced to prevent.
    //
    // Coverage note: the tiny-llm fixture emits `tok22tok22tok20<eos>` with no
    // trailing `\n`, so `needs_trailing_newline` is always `true` here and the
    // "skip when already ends with \n" branch is never exercised by this test.
    // That branch — and the predicate as a whole — is covered by the unit tests
    // in `output.rs::tests::tty_trailing_newline_predicate_covers_all_cases`.
    // This test is the integration proof that the "add" path is wired to a real
    // terminal; it catches a "never add" regression (assertion on `ends_with
    // \r\n` would fail).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn tty_stdout_gets_exactly_one_trailing_newline_after_streaming() {
        let (master, slave) = open_pty();
        let slave_stdin = dup(&slave).expect("dup must succeed");
        let slave_stdout = slave;

        let child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
            .arg("generate")
            .arg(text_model())
            .arg("--raw")
            .arg("--stream") // activates the conditional-newline path in run_generation_turn
            .arg("--prompt")
            .arg("hi")
            .arg("--max-new-tokens")
            .arg("5")
            .env("ONNX_GENAI_EP", "cpu")
            .stdin(Stdio::from(slave_stdin))
            .stdout(Stdio::from(slave_stdout)) // TTY → conditional newline branch runs
            .stderr(Stdio::null())
            .spawn()
            .expect("CLI binary must start");

        // Drain concurrently so the child never blocks writing to the PTY.
        let drain = std::thread::spawn(move || drain_pty_master(master));
        let _ = child.wait_with_output().expect("child must exit cleanly");
        let pty_bytes = drain.join().expect("drain thread must finish");

        let pty_out = String::from_utf8_lossy(&pty_bytes);

        // ONLCR converts \n → \r\n, so the final separator must be \r\n.
        assert!(
            pty_out.ends_with("\r\n"),
            "TTY stdout must end with a single CRLF after the streamed reply; \
             got: {pty_out:?}"
        );

        // A double CRLF would mean an extra blank line was injected — the
        // defect the conditional newline logic prevents.
        assert!(
            !pty_out.ends_with("\r\n\r\n"),
            "TTY stdout must not end with a blank line (double CRLF); \
             the reply already ended with \\n and an extra newline was added. \
             got: {pty_out:?}"
        );
    }
}
