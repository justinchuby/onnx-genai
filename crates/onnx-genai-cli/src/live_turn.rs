//! Live rendering of a generating turn: the reply and its numbers, drawn
//! together.
//!
//! Generation numbers are most useful while they are moving — a reader watching
//! tokens appear wants to see throughput settle, time to first token land, and
//! whether the prompt was reused or recomputed. Printing them afterwards answers
//! the question too late.
//!
//! The reply and the status line are therefore drawn as **one** frame rather
//! than interleaving raw prints with a separately-positioned bar. A single
//! buffer diffed per frame is what keeps the display from tearing: nothing is
//! ever half-written, and no line has to migrate between two rendering paths
//! while the eye is on it.
//!
//! # Scrollback is kept
//!
//! This uses ratatui's *inline* viewport, not an alternate screen. Only the last
//! few lines of the running reply live in the viewport; as it grows, finished
//! lines spill upward into the terminal's own scrollback, exactly where ordinary
//! output would have gone. The conversation stays selectable, copyable, and
//! present after the session ends — all of which a full-screen TUI would take
//! away.
//!
//! # Not a terminal, no rendering
//!
//! Everything here is inert unless stdout is a terminal. A piped or redirected
//! run gets exactly the plain text it got before, with no escape sequences and
//! no reserved lines. That fallback is not a courtesy: the REPL's whole test
//! suite drives it over a pipe.
//!
//! The terminal is never put into raw mode. The REPL reads whole lines from
//! stdin, which depends on the line discipline raw mode removes.

use std::io::{self, IsTerminal, Write};

use ratatui::Terminal;
use ratatui::layout::Rect;
use ratatui::prelude::{CrosstermBackend, Line, Span, Style};
use ratatui::widgets::{Paragraph, Widget, Wrap};
use ratatui::{TerminalOptions, Viewport};

/// Rows the live view occupies, including the status line.
///
/// Small enough to read as a status area rather than a pager, large enough that
/// a few lines of reply are visible while they are written.
const VIEWPORT_ROWS: u16 = 8;

/// Renders a generating turn, or writes straight through when there is no
/// terminal.
pub(crate) enum LiveTurn {
    /// A terminal is present but nothing has been rendered yet.
    ///
    /// Initialization is deferred because setting up an inline viewport asks the
    /// terminal where the cursor is (`ESC[6n`) and *reads stdin* for the reply.
    /// Doing that while the REPL is waiting on a typed line lets the setup
    /// swallow what the user typed. By the time anything is rendered the line
    /// has already been consumed by the REPL's own read, so the query is safe.
    Pending,
    Active(Box<Active>),
    Disabled,
}

pub(crate) struct Active {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    /// Logical lines of the reply still shown in the viewport, each a run of
    /// `(text, is_reasoning)` segments. Earlier lines have already spilled into
    /// scrollback.
    ///
    /// Segments rather than plain strings because a model's thinking is dimmed
    /// to set it apart from its answer, and styling has to survive re-rendering
    /// — writing an escape code into the stream would be overwritten by the next
    /// frame.
    lines: Vec<Vec<Segment>>,
    status: String,
}

/// A run of reply text with one style.
#[derive(Clone, Debug)]
pub(crate) struct Segment {
    text: String,
    reasoning: bool,
}

impl LiveTurn {
    /// Reserve the live area, or return [`LiveTurn::Disabled`] when stdout is
    /// not a terminal.
    ///
    /// A terminal that refuses to initialize falls back to plain output rather
    /// than ending the session: that fallback is already a supported mode.
    pub(crate) fn new() -> Self {
        if io::stdout().is_terminal() {
            Self::Pending
        } else {
            Self::Disabled
        }
    }

    /// Whether this will render, which is what tells the caller not to emit its
    /// own newlines and spacing.
    pub(crate) fn is_active(&self) -> bool {
        matches!(self, Self::Pending | Self::Active(_))
    }

    /// Claim the live area on first use, degrading to plain output if the
    /// terminal will not give it up.
    fn activate(&mut self) -> Option<&mut Active> {
        if matches!(self, Self::Pending) {
            let options = TerminalOptions {
                viewport: Viewport::Inline(VIEWPORT_ROWS),
            };
            *self = match Terminal::with_options(CrosstermBackend::new(io::stdout()), options) {
                Ok(terminal) => Self::Active(Box::new(Active {
                    terminal,
                    lines: vec![Vec::new()],
                    status: String::new(),
                })),
                Err(_) => Self::Disabled,
            };
        }
        match self {
            Self::Active(active) => Some(active),
            _ => None,
        }
    }

    /// Append generated text. `reasoning` marks the model's thinking, which is
    /// dimmed to set it apart from the answer.
    ///
    /// This intentionally does **not** redraw. The decode loop can produce
    /// tokens far faster than a terminal can repaint; callers coalesce draws on
    /// a frame timer and call [`draw`](Self::draw) when a frame is due.
    pub(crate) fn push(&mut self, text: &str, reasoning: bool) -> anyhow::Result<()> {
        match self.activate() {
            Some(active) => active.push(text, reasoning),
            None => {
                print!("{text}");
                io::stdout().flush()?;
                Ok(())
            }
        }
    }

    /// Replace the numbers shown beneath the reply.
    pub(crate) fn set_status(&mut self, status: String) -> anyhow::Result<()> {
        // Never activates on its own: a status with no reply to sit under would
        // claim the terminal for nothing.
        match self {
            Self::Active(active) => {
                active.status = status;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Redraw the coalesced reply/status frame if the viewport is active.
    pub(crate) fn draw(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Active(active) => {
                active.spill()?;
                active.draw()
            }
            _ => Ok(()),
        }
    }

    /// End the turn: spill the whole reply into scrollback and clear the area.
    pub(crate) fn finish(&mut self, output_needs_trailing_newline: bool) -> anyhow::Result<()> {
        match self {
            Self::Active(active) => active.finish(output_needs_trailing_newline),
            // Nothing was rendered, so any TTY separator still belongs to the plain path.
            _ => {
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
        for (index, chunk) in text.split('\n').enumerate() {
            if index > 0 {
                self.lines.push(Vec::new());
            }
            if chunk.is_empty() {
                continue;
            }
            let line = self.lines.last_mut().expect("at least one line");
            // Extend the trailing segment when the style is unchanged, so a
            // token-by-token stream does not accumulate one segment per token.
            match line.last_mut() {
                Some(last) if last.reasoning == reasoning => last.text.push_str(chunk),
                _ => line.push(Segment {
                    text: chunk.to_string(),
                    reasoning,
                }),
            }
        }
        Ok(())
    }

    /// Width available to the reply, never zero so the wrap arithmetic stays
    /// meaningful on a degenerate terminal.
    fn width(&self) -> usize {
        self.terminal
            .size()
            .map_or(80, |size| size.width as usize)
            .max(1)
    }

    /// Move finished lines into scrollback until the reply fits the live area.
    ///
    /// The status line owns the bottom row, so the reply gets the rest.
    fn spill(&mut self) -> anyhow::Result<()> {
        let width = self.width();
        let budget = (VIEWPORT_ROWS as usize).saturating_sub(1).max(1);
        while self.lines.len() > 1 && wrapped_rows(&self.lines, width) > budget {
            let line = self.lines.remove(0);
            self.commit(line, width)?;
        }
        Ok(())
    }

    /// Write one finished line above the live area, into the terminal's own
    /// scrollback.
    fn commit(&mut self, line: Vec<Segment>, width: usize) -> anyhow::Result<()> {
        let rows = wrapped_rows(std::slice::from_ref(&line), width) as u16;
        self.terminal.insert_before(rows, |buffer| {
            Paragraph::new(render_line(&line))
                .wrap(Wrap { trim: false })
                .render(buffer.area, buffer);
        })?;
        Ok(())
    }

    fn draw(&mut self) -> anyhow::Result<()> {
        let lines: Vec<Line> = self.lines.iter().map(|line| render_line(line)).collect();
        let status = self.status.clone();
        self.terminal.draw(|frame| {
            let area = frame.area();
            let reply_height = area.height.saturating_sub(1);
            let reply_area = Rect {
                height: reply_height,
                ..area
            };
            let status_area = Rect {
                y: area.y + reply_height,
                height: 1,
                ..area
            };
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(reply_area, frame.buffer_mut());
            status_widget(&status).render(status_area, frame.buffer_mut());
        })?;
        Ok(())
    }

    fn finish(&mut self, output_needs_trailing_newline: bool) -> anyhow::Result<()> {
        if output_needs_trailing_newline {
            self.lines.push(Vec::new());
        }
        let width = self.width();
        for line in std::mem::take(&mut self.lines) {
            self.commit(line, width)?;
        }
        self.lines = vec![Vec::new()];
        self.status.clear();
        // Leave the area blank rather than holding a finished turn's numbers
        // next to the prompt the user is about to type into.
        self.terminal.draw(|frame| {
            Paragraph::new("").render(frame.area(), frame.buffer_mut());
        })?;
        Ok(())
    }
}

/// Rows `lines` occupy once wrapped to `width`.
///
/// Count through ratatui's own wrapped-line composer rather than duplicating a
/// display-width approximation. `Paragraph::wrap` is word-wrap aware, so a long
/// word after a short word can occupy more rows than `ceil(total_width / width)`;
/// undercounting here makes the inline viewport reserve too few rows and redraw
/// over terminal scrollback.
fn wrapped_rows(lines: &[Vec<Segment>], width: usize) -> usize {
    Paragraph::new(
        lines
            .iter()
            .map(|line| render_line(line))
            .collect::<Vec<_>>(),
    )
    .wrap(Wrap { trim: false })
    .line_count(width.min(u16::MAX as usize).max(1) as u16)
}

/// Style a reply line, dimming the model's thinking.
fn render_line(line: &[Segment]) -> Line<'static> {
    Line::from(
        line.iter()
            .map(|segment| {
                let style = if segment.reasoning {
                    Style::new().add_modifier(ratatui::style::Modifier::DIM)
                } else {
                    Style::new()
                };
                Span::styled(segment.text.clone(), style)
            })
            .collect::<Vec<_>>(),
    )
}

/// The status line, dimmed so it reads as chrome rather than as model output.
fn status_widget(status: &str) -> Paragraph<'_> {
    Paragraph::new(Line::from(Span::styled(
        status,
        Style::new().add_modifier(ratatui::style::Modifier::DIM),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> Vec<Segment> {
        if text.is_empty() {
            return Vec::new();
        }
        vec![Segment {
            text: text.to_string(),
            reasoning: false,
        }]
    }

    #[test]
    fn an_empty_line_still_occupies_a_row() {
        assert_eq!(wrapped_rows(&[line("")], 20), 1);
    }

    #[test]
    fn a_line_wider_than_the_terminal_occupies_several_rows() {
        assert_eq!(wrapped_rows(&[line(&"a".repeat(45))], 20), 3);
        assert_eq!(wrapped_rows(&[line(&"a".repeat(40))], 20), 2);
        assert_eq!(wrapped_rows(&[line(&"a".repeat(20))], 20), 1);
    }

    #[test]
    fn rows_are_summed_across_lines() {
        assert_eq!(
            wrapped_rows(&[line(&"a".repeat(25)), line("b"), line("")], 20),
            4
        );
    }

    #[test]
    fn a_wrapped_row_count_spans_the_segments_of_one_line() {
        // Styling splits a line into segments; wrapping does not care where the
        // splits fall, only how wide the line is in total.
        let mixed = vec![
            Segment {
                text: "a".repeat(15),
                reasoning: true,
            },
            Segment {
                text: "b".repeat(15),
                reasoning: false,
            },
        ];
        assert_eq!(wrapped_rows(&[mixed], 20), 2);
    }

    #[test]
    fn row_count_matches_ratatui_word_wrap_not_total_width_ceiling() {
        // At width 10, ratatui word-wraps this as:
        //   "aaa "
        //   "bbbbbbbbbb"
        //   "bbbbb"
        // A naive `ceil(total_display_width / width)` reservation sees only
        // 19 columns and reserves 2 rows, which is exactly how the live viewport
        // under-reserved and overpainted existing scrollback.
        let text = format!("aaa {}", "b".repeat(15));
        let old_reserved_rows = text.len().div_ceil(10);
        assert_eq!(old_reserved_rows, 2);
        assert_eq!(wrapped_rows(&[line(&text)], 10), 3);
    }

    #[test]
    fn wide_characters_occupy_two_columns_each() {
        // Each CJK ideograph renders in two terminal cells, so 15 of them are 30
        // columns wide and wrap to two rows in a 20-column terminal. Counting
        // scalar values instead would see 15 <= 20 and wrongly report one row —
        // the miscount behind the garbled Chinese REPL output.
        let cjk = "字".repeat(15);
        assert_eq!(wrapped_rows(&[line(&cjk)], 20), 2);
        // Exactly fills two rows: 20 ideographs == 40 columns == 2 * 20.
        assert_eq!(wrapped_rows(&[line(&"字".repeat(20))], 20), 2);
        // 21 ideographs == 42 columns spills into a third row.
        assert_eq!(wrapped_rows(&[line(&"字".repeat(21))], 20), 3);
    }

    #[test]
    fn mixed_ascii_and_wide_text_sums_display_width_across_segments() {
        // "abc" (3 columns) + 10 ideographs (20 columns) == 23 columns, which
        // wraps to two rows at width 20. Char counting would see 13 and report
        // one row. The wide run is a separate segment to prove the width sums
        // correctly across a styling split.
        let mixed = vec![
            Segment {
                text: "abc".to_string(),
                reasoning: false,
            },
            Segment {
                text: "文".repeat(10),
                reasoning: true,
            },
        ];
        assert_eq!(wrapped_rows(&[mixed], 20), 2);
    }

    #[test]
    fn a_combining_mark_adds_no_width() {
        // "e" + U+0301 (combining acute) is one column, not two: zero-width
        // marks must not inflate the row math. This documents the general
        // display-width fix rather than a CJK special case.
        let combined = "e\u{0301}".repeat(20);
        assert_eq!(wrapped_rows(&[line(&combined)], 20), 1);
    }

    #[test]
    fn thinking_is_dimmed_and_the_answer_is_not() {
        let rendered = render_line(&[
            Segment {
                text: "thinking".to_string(),
                reasoning: true,
            },
            Segment {
                text: "answer".to_string(),
                reasoning: false,
            },
        ]);
        assert!(
            rendered.spans[0]
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::DIM)
        );
        assert!(
            !rendered.spans[1]
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::DIM)
        );
    }

    /// Without a terminal nothing may be reserved or escaped, because the REPL's
    /// tests — and any script or captured transcript — read the plain text.
    #[test]
    fn a_pipe_gets_the_plain_text_path() {
        // Tests do not run attached to a terminal, so this is the real path.
        let mut live = LiveTurn::new();
        assert!(!live.is_active());
        assert!(matches!(live, LiveTurn::Disabled));
        assert!(live.activate().is_none(), "a pipe must never be claimed");
    }

    /// Setting up an inline viewport reads stdin for the terminal's cursor
    /// reply, so it must not happen while the REPL is waiting on a typed line.
    #[test]
    fn a_terminal_is_not_claimed_before_anything_is_rendered() {
        let mut pending = LiveTurn::Pending;
        pending
            .set_status("42 tok/s".to_string())
            .expect("a status alone is not worth claiming the terminal for");
        assert!(
            matches!(pending, LiveTurn::Pending),
            "status updates must not trigger setup"
        );
    }
}
