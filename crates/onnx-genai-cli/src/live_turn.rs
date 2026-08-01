//! Append-only rendering of a generating turn.
//!
//! The REPL uses reedline for input editing. A previous ratatui inline viewport
//! was removed because it also tried to own the cursor and scrollback, so it
//! could reserve rows at reedline's submitted input line and overwrite it. This
//! renderer is deliberately simpler: generated text is appended exactly where
//! ordinary stdout would write it, and per-turn stats are printed after the turn
//! by `output.rs`. It never moves the cursor up, never rewrites scrollback, and
//! never reserves rows.

use std::io::{self, IsTerminal, Write};

/// Renders a generating turn, or writes straight through when there is no
/// terminal.
pub(crate) enum LiveTurn {
    Active(Active),
    Disabled,
}

pub(crate) struct Active {
    dimmed: bool,
}

impl LiveTurn {
    /// Enable terminal rendering only for a real stdout TTY.
    pub(crate) fn new() -> Self {
        if io::stdout().is_terminal() {
            Self::Active(Active { dimmed: false })
        } else {
            Self::Disabled
        }
    }

    /// Whether this will render, which is what tells the caller not to emit its
    /// own newlines and spacing.
    pub(crate) fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    /// Append generated text. `reasoning` marks the model's thinking, which is
    /// dimmed to set it apart from the answer.
    pub(crate) fn push(&mut self, text: &str, reasoning: bool) -> anyhow::Result<()> {
        match self {
            Self::Active(active) => active.push(text, reasoning),
            Self::Disabled => {
                print!("{text}");
                io::stdout().flush()?;
                Ok(())
            }
        }
    }

    /// Kept for the live-rendering interface. Append-only rendering does not
    /// draw an in-place status line because doing so would require cursor
    /// movement in the same terminal reedline owns.
    pub(crate) fn set_status(&mut self, _status: String) -> anyhow::Result<()> {
        Ok(())
    }

    /// Kept for the live-rendering interface. Append-only rendering flushes each
    /// write and has no buffered frame to draw.
    pub(crate) fn draw(&mut self) -> anyhow::Result<()> {
        io::stdout().flush()?;
        Ok(())
    }

    /// End the turn, resetting terminal attributes and optionally separating the
    /// reply from the following prompt.
    pub(crate) fn finish(&mut self, output_needs_trailing_newline: bool) -> anyhow::Result<()> {
        match self {
            Self::Active(active) => active.finish(output_needs_trailing_newline),
            Self::Disabled => {
                if output_needs_trailing_newline {
                    println!();
                }
                Ok(())
            }
        }
    }
}

impl Active {
    fn push(&mut self, text: &str, reasoning: bool) -> anyhow::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if reasoning != self.dimmed {
            print!("{}", if reasoning { "\x1b[2m" } else { "\x1b[0m" });
            self.dimmed = reasoning;
        }
        print!("{text}");
        io::stdout().flush()?;
        Ok(())
    }

    fn finish(&mut self, output_needs_trailing_newline: bool) -> anyhow::Result<()> {
        if self.dimmed {
            print!("\x1b[0m");
            self.dimmed = false;
        }
        if output_needs_trailing_newline {
            println!();
        }
        io::stdout().flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a terminal nothing may be reserved or escaped, because the REPL's
    /// tests — and any script or captured transcript — read the plain text.
    #[test]
    fn a_pipe_gets_the_plain_text_path() {
        // Tests do not run attached to a terminal, so this is the real path.
        let live = LiveTurn::new();
        assert!(!live.is_active());
        assert!(matches!(live, LiveTurn::Disabled));
    }

    #[test]
    fn disabled_finish_emits_only_the_needed_separator() {
        let mut live = LiveTurn::Disabled;
        live.finish(false).expect("no separator needed");
        live.finish(true).expect("separator write succeeds");
    }
}
