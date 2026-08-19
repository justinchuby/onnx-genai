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
    use std::io::{Read, Write};
    use std::os::fd::OwnedFd;
    use std::os::unix::io::AsFd;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    use nix::pty::{OpenptyResult, Winsize, openpty};
    use nix::unistd::dup;

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn text_model() -> PathBuf {
        repository_root().join("tests/fixtures/tiny-llm")
    }

    fn strip_terminal_controls(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if let Some('[') = chars.next() {
                    for next in chars.by_ref() {
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else if ch != '\r' {
                out.push(ch);
            }
        }
        out
    }

    fn text_has_nearby_ansi_style(raw: &str, text: &str) -> bool {
        raw.match_indices(text).any(|(index, _)| {
            let window_start = index.saturating_sub(32);
            raw[window_start..index].contains("\x1b[")
        })
    }

    /// Open a fresh PTY pair.  The slave is the child-side terminal device;
    /// the master is the parent-side pipe into/out of that device.
    ///
    /// The slave is given a real 24×80 window size on purpose — **do not pass
    ///   `None` here.**  `openpty(None, ..)` leaves the window at 0×0. The
    /// append-only `run` renderer no longer depends on a viewport, but reedline
    /// still renders an interactive prompt and needs a realistic terminal size
    /// for wrapping/repaint behavior. 24×80 is the smallest ordinary terminal
    /// size that avoids false terminal-harness failures; the exact numbers are
    /// not magic, only "non-zero and realistic".
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

    /// How long `drain_pty_master` waits **for the next byte** before it gives
    /// up.  This is an *idle* timeout, reset on every byte received — not a
    /// total budget — and that distinction is deliberate.
    ///
    /// Two regimes this harness must tolerate without either hanging CI or
    /// flaking on a false content failure:
    ///
    /// * **Slow start.**  The child writes nothing to *stdout* until its first
    ///   token, because the model must load first (its progress logs go to
    ///   stderr, not the PTY).  On the slowest real machine measured for this
    ///   suite — a debug build over WSL 2's `/mnt/c` under parallel test load —
    ///   cold load-to-first-token was ~48 s.  A previous **30 s total** budget
    ///   lost to exactly that and returned an *empty* read, which then tripped
    ///   the trailing-newline assertion and masqueraded as a rendering defect.
    ///   120 s is ~2.5× that measured worst case: headroom for a cold CI runner
    ///   with a cold cargo/model cache, not a number picked to feel safe.
    /// * **Long stream.**  A future test that drives the `run` REPL against a
    ///   real, many-token reply streams for a long time.  Because the clock is
    ///   reset on every byte, an arbitrarily long reply never trips the timeout
    ///   as long as tokens keep arriving; only a genuinely *stuck* child (no
    ///   byte for the whole idle window) fails — and it fails promptly instead
    ///   of hanging the runner for the job's hard limit.
    const DRAIN_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

    /// Outcome of draining a PTY master.
    ///
    /// `timed_out` distinguishes the two ways draining can end so a caller can
    /// report them differently (RULES.md rule 1): `false` means the child
    /// closed the slave (clean EOF/EIO — draining is complete), `true` means no
    /// byte arrived for `DRAIN_IDLE_TIMEOUT` (a stuck or too-slow child).
    /// Callers **must** check `timed_out` before asserting on `bytes`, so a
    /// "the child stopped producing output" failure never masquerades as a
    /// "the content was wrong" failure.
    struct DrainOutcome {
        bytes: Vec<u8>,
        timed_out: bool,
    }

    /// Drain bytes from a PTY master until the child closes the slave (clean
    /// EOF/EIO) or no byte arrives for `DRAIN_IDLE_TIMEOUT` (a stuck child).
    ///
    /// Uses `poll()` with the idle timeout instead of a blocking `read()` so a
    /// child that never writes cannot block the test runner forever.  Each
    /// `poll()` waits up to the full idle window for the *next* byte, so the
    /// clock is effectively reset on every successful read: a slow start or a
    /// long stream is tolerated, while a genuinely stuck child still fails.
    fn drain_pty_master(master: OwnedFd) -> DrainOutcome {
        let mut file: std::fs::File = master.into();
        let mut buf = [0u8; 4096];
        let mut collected = Vec::new();
        loop {
            let ready = {
                let borrowed = file.as_fd();
                let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
                poll(
                    &mut fds,
                    PollTimeout::try_from(DRAIN_IDLE_TIMEOUT).unwrap_or(PollTimeout::MAX),
                )
                .unwrap_or(0)
            };
            if ready == 0 {
                // No byte arrived within the idle window: a stuck/too-slow
                // child.  Return what we have and flag it so the caller reports
                // a timeout, not a content mismatch.
                return DrainOutcome {
                    bytes: collected,
                    timed_out: true,
                };
            }
            match file.read(&mut buf) {
                Ok(0) => break, // EOF: slave closed cleanly.
                Ok(n) => collected.extend_from_slice(&buf[..n]),
                // EIO when the last holder of the slave closes it (child exits).
                Err(_) => break,
            }
        }
        DrainOutcome {
            bytes: collected,
            timed_out: false,
        }
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
    #[ignore = "flaky under CI load, see #615: the drain loop can consume the \
                full 120s idle window after the child has already produced \
                everything the test asserts on. Re-enable with the fix, not by \
                raising the timeout."]
    fn tty_stdout_gets_exactly_one_trailing_newline_after_streaming() {
        let (master, slave) = open_pty();
        let slave_stdin = dup(&slave).expect("dup must succeed");
        let slave_stdout = slave;

        let child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
            .arg("generate")
            .arg(text_model())
            .arg("--raw")
            .arg("--stream") // activates the conditional-newline path in run_generation_turn
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
        let outcome = drain.join().expect("drain thread must finish");

        // Separate "the child stopped producing output" from "the content was
        // wrong" (RULES.md rule 1).  A timeout here is NOT a trailing-newline
        // defect, and reporting it as one would send whoever triages it hunting
        // a rendering bug that isn't there.  Check it first, with an actionable
        // message, so the two failures can never be confused.
        assert!(
            !outcome.timed_out,
            "timed out after {}s with no new bytes from the PTY master while \
             waiting for the streamed reply ({} byte(s) collected so far). This \
             is a stuck or too-slow child, NOT a trailing-newline defect. What \
             to check, in order: (1) the `tests/fixtures/tiny-llm` model still \
             loads and streams under `generate --stream`; (2) the machine is \
             not so loaded that cold load-to-first-token exceeds the {}s idle \
             budget (`DRAIN_IDLE_TIMEOUT`) — raise it if a slower baseline is \
             now normal; (3) only after ruling those out, suspect the newline \
             logic. Bytes so far: {:?}",
            DRAIN_IDLE_TIMEOUT.as_secs(),
            outcome.bytes.len(),
            DRAIN_IDLE_TIMEOUT.as_secs(),
            String::from_utf8_lossy(&outcome.bytes),
        );

        let pty_out = String::from_utf8_lossy(&outcome.bytes);

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

    #[test]
    #[ignore = "flaky under CI load, see #615: the turn state machine counts \
                '>>>' occurrences, which reedline's prompt repaints inflate \
                non-deterministically. Re-enable once it advances on a \
                once-per-turn event instead."]
    fn run_repl_preserves_submitted_lines_across_two_tty_turns() {
        let (master, slave) = open_pty();
        let slave_stdin = dup(&slave).expect("dup stdin must succeed");
        let slave_stdout = dup(&slave).expect("dup stdout must succeed");
        let slave_stderr = slave;

        let mut child = Command::new(env!("CARGO_BIN_EXE_onnx-genai"))
            .arg("run")
            .arg(text_model())
            .arg("--raw")
            .arg("--max-new-tokens")
            .arg("2")
            .env("ONNX_GENAI_EP", "cpu")
            .stdin(Stdio::from(slave_stdin))
            .stdout(Stdio::from(slave_stdout))
            .stderr(Stdio::from(slave_stderr))
            .spawn()
            .expect("CLI binary must start");

        let mut master: std::fs::File = master.into();
        let mut bytes = Vec::new();
        let mut first_sent = false;
        let mut second_sent = false;
        let mut exit_sent = false;
        let mut dsr_answered = 0usize;
        let mut buf = [0u8; 4096];
        loop {
            let ready = {
                let borrowed = master.as_fd();
                let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
                poll(
                    &mut fds,
                    PollTimeout::try_from(DRAIN_IDLE_TIMEOUT).unwrap_or(PollTimeout::MAX),
                )
                .unwrap_or(0)
            };
            assert!(
                ready != 0,
                "timed out waiting for two-turn REPL transcript; bytes so far: {:?}",
                String::from_utf8_lossy(&bytes)
            );

            match master.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => bytes.extend_from_slice(&buf[..n]),
                Err(_) => break,
            }

            // reedline probes the cursor position (DSR, `ESC[6n`) while setting
            // up its line editor; a real terminal answers with `ESC[<row>;<col>R`.
            // This headless PTY has no emulator behind it, so the harness must
            // answer or reedline aborts the turn with "The cursor position could
            // not be read within a normal duration". Reply once per query.
            const DSR_QUERY: &[u8] = b"\x1b[6n";
            let dsr_seen = bytes
                .windows(DSR_QUERY.len())
                .filter(|w| *w == DSR_QUERY)
                .count();
            while dsr_answered < dsr_seen {
                master
                    .write_all(b"\x1b[1;1R")
                    .expect("answer cursor-position query");
                dsr_answered += 1;
            }

            let transcript = strip_terminal_controls(&String::from_utf8_lossy(&bytes));
            let prompt_count = transcript.matches(">>>").count();
            if !first_sent && prompt_count >= 1 {
                master.write_all(b"first\r").expect("send first prompt");
                first_sent = true;
            }
            if first_sent && !second_sent && prompt_count >= 2 {
                master.write_all(b"second\r").expect("send second prompt");
                second_sent = true;
            }
            if second_sent && !exit_sent && prompt_count >= 3 {
                master.write_all(b"\r").expect("send empty exit line");
                exit_sent = true;
            }
            if exit_sent && child.try_wait().expect("poll child").is_some() {
                break;
            }
        }
        let _ = child.wait();

        let transcript = strip_terminal_controls(&String::from_utf8_lossy(&bytes));
        let raw = String::from_utf8_lossy(&bytes);
        assert!(
            text_has_nearby_ansi_style(&raw, "first") && text_has_nearby_ansi_style(&raw, "second"),
            "TTY reedline input should be styled distinctly from model output; raw transcript: {raw:?}"
        );
        let first_input = transcript
            .find(">>> first")
            .unwrap_or_else(|| panic!("first submitted line was not preserved: {transcript:?}"));
        let second_input = transcript
            .find(">>> second")
            .unwrap_or_else(|| panic!("second submitted line was not preserved: {transcript:?}"));
        let first_answer = transcript[first_input..second_input]
            .find("tok")
            .map(|offset| first_input + offset)
            .unwrap_or_else(|| {
                panic!("first reply text missing before second input: {transcript:?}")
            });
        let second_answer = transcript[second_input..]
            .find("tok")
            .map(|offset| second_input + offset)
            .unwrap_or_else(|| {
                panic!("second reply text missing after second input: {transcript:?}")
            });

        assert!(
            first_input < first_answer
                && first_answer < second_input
                && second_input < second_answer,
            "turn ordering must be input1 -> reply1 -> input2 -> reply2; transcript: {transcript:?}"
        );
        for line in transcript.lines() {
            assert!(
                !(line.trim().is_empty() && line.len() >= 8),
                "renderer leaked a blank/whitespace row into scrollback: {transcript:?}"
            );
        }
    }
}
